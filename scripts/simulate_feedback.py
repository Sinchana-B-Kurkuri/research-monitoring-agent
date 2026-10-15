"""
simulate_feedback.py
~~~~~~~~~~~~~~~~~~~~
CLI for Simulated Feedback Generation & Demo Database Snapshot Preparation (Roadmap Sections 8.4 & 14).

Usage:
  # Apply oracle feedback to pending review tasks and delivered digest items for a user
  python scripts/simulate_feedback.py --user-id alice --cycle-id 1

  # Prepare a complete demo database snapshot with prerecorded simulated feedback for Phase 14
  python scripts/simulate_feedback.py --prepare-demo-snapshot
"""

import argparse
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import sys

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pandas as pd

from research_monitor.config import config
from research_monitor.database.models import (
    Base,
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    InterestProfile,
    InterestVectorVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    User,
    create_training_labels_view,
)
from research_monitor.database.session import SessionLocal, init_db
from research_monitor.evaluation.labeling import load_latest_labels
from research_monitor.evaluation.simulation import SimulationOracle
from research_monitor.relevance.state import DecisionStatus
from research_monitor.review.digest import resume_after_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def apply_oracle_feedback(
    user_id: str,
    cycle_id: Optional[int] = None,
    noise_rate: float = 0.0,
    session=None,
) -> dict[str, int]:
    """
    Applies oracle decisions to all pending review tasks and unreacted digest items
    in a given monitoring cycle (or latest cycle).
    """
    owns_session = False
    if session is None:
        init_db()
        session = SessionLocal()
        owns_session = True

    try:
        # Load human ground-truth labels if available as ground-truth oracle map
        labels_df = load_latest_labels()
        gt_map = {}
        if labels_df is not None and len(labels_df) > 0:
            gt_map = dict(zip(labels_df["paper_id"].astype(str), labels_df["label"].astype(str)))

        oracle = SimulationOracle(ground_truth_map=gt_map, noise_rate=noise_rate)

        # 1. Resolve target cycle
        query = session.query(Cycle).filter_by(user_id=user_id)
        if cycle_id is not None:
            cycle = query.filter_by(id=cycle_id).first()
        else:
            cycle = query.order_by(Cycle.id.desc()).first()

        if cycle is None:
            logger.warning("No cycle found for user '%s'.", user_id)
            return {"reviewer_events": 0, "user_events": 0}

        target_cycle_id = cycle.id
        now = datetime.now(timezone.utc)

        # 2. Process pending review tasks (Borderline escalation)
        pending_tasks = (
            session.query(ReviewTask)
            .join(RoutingDecision, ReviewTask.decision_id == RoutingDecision.id)
            .filter(
                RoutingDecision.user_id == user_id,
                RoutingDecision.cycle_id == target_cycle_id,
                ReviewTask.decision.is_(None),
            )
            .all()
        )

        reviewer_count = 0
        for task in pending_tasks:
            dec = task.routing_decision
            paper = dec.paper if dec else None
            meta = {"title": paper.title, "abstract": paper.abstract} if paper else None
            oracle_dec = oracle.get_reviewer_decision(dec.paper_id, meta)
            
            task.decision = oracle_dec
            task.decided_at = now

            # Emit reviewer feedback event
            lbl_val = 1 if oracle_dec == "accept" else 0
            fe = FeedbackEvent(
                user_id=user_id,
                paper_id=dec.paper_id,
                source="reviewer",
                label=lbl_val,
                decision_id=dec.id,
                created_at=now,
            )
            session.add(fe)
            reviewer_count += 1

        session.commit()

        # 3. Resume workflow to deliver accepted papers to digest
        resume_after_review(user_id=user_id, cycle_id=target_cycle_id, session=session)

        # 4. Process unreacted digest items (User reactions)
        digest = (
            session.query(Digest)
            .filter_by(user_id=user_id, cycle_id=target_cycle_id)
            .order_by(Digest.id.desc())
            .first()
        )

        user_count = 0
        if digest is not None:
            for item in digest.items:
                dec = item.routing_decision
                if dec.status in (DecisionStatus.DELIVERED.value, DecisionStatus.ROUTED_HIGH.value):
                    paper = dec.paper if dec else None
                    meta = {"title": paper.title, "abstract": paper.abstract} if paper else None
                    user_reaction = oracle.get_user_reaction(dec.paper_id, meta)

                    fe = FeedbackEvent(
                        user_id=user_id,
                        paper_id=dec.paper_id,
                        source="user",
                        label=user_reaction,
                        decision_id=dec.id,
                        created_at=now,
                    )
                    session.add(fe)
                    dec.status = DecisionStatus.REACTED.value
                    user_count += 1

            session.commit()

        logger.info(
            "Oracle feedback applied to cycle #%d: %d reviewer decisions, %d user reactions",
            target_cycle_id,
            reviewer_count,
            user_count,
        )
        return {"reviewer_events": reviewer_count, "user_events": user_count}

    finally:
        if owns_session:
            session.close()


def prepare_demo_snapshot(session=None) -> str:
    """
    Prepares a demo database snapshot with a simulated researcher ('simulated_researcher')
    containing 3 completed replay cycles and 40+ simulated feedback events.
    Enables Phase 14 live retraining demonstration with one command.
    """
    owns_session = False
    if session is None:
        init_db()
        session = SessionLocal()
        owns_session = True

    try:
        user_id = "simulated_researcher"
        now = datetime.now(timezone.utc)

        # Check if user already exists
        user = session.query(User).filter_by(id=user_id).first()
        if user is None:
            user = User(
                id=user_id,
                name="Dr. Alex Rivera (Simulated)",
                access_token="demo_token_alex_rivera_777",
                is_simulated=True,
                created_at=now - timedelta(days=5),
            )
            session.add(user)
            session.flush()

        # Seed interest embedding
        profile = session.query(InterestProfile).filter_by(user_id=user.id).first()
        if profile is None or profile.seed_embedding is None:
            # Create profile
            rng = np.random.default_rng(42)
            dummy_vec = rng.standard_normal(384).astype(np.float32)
            dummy_vec /= np.linalg.norm(dummy_vec)
            if profile is None:
                profile = InterestProfile(
                    user_id=user.id,
                    seed_text="AI agents, autonomous systems, reinforcement learning, LLM tool use, and multi-agent planning.",
                    seed_embedding=dummy_vec.tolist(),
                    created_at=now - timedelta(days=5),
                )
                session.add(profile)
            else:
                profile.seed_embedding = dummy_vec.tolist()
            session.flush()
            seed_emb = dummy_vec
        else:
            seed_emb = np.array(profile.seed_embedding, dtype=np.float32)

        # Ensure v0 active vector version
        v0 = session.query(InterestVectorVersion).filter_by(user_id=user.id, version=0).first()
        if v0 is None:
            v0 = InterestVectorVersion(
                user_id=user.id,
                version=0,
                vector=seed_emb.tolist() if hasattr(seed_emb, "tolist") else list(seed_emb),
                method="seed",
                n_effective_labels=0,
                p_high_level=98.0,
                p_low_level=90.0,
                t_high=0.75,
                t_low=0.55,
                created_at=now - timedelta(days=5),
                is_active=True,
            )
            session.add(v0)
            session.flush()

        # Create 40 synthetic/replay papers and feedback events (20 pos, 20 neg)
        rng = np.random.default_rng(101)
        seed_norm = np.array(seed_emb) / np.linalg.norm(seed_emb)

        # Create cycle
        cycle = Cycle(
            user_id=user.id,
            as_of_date="2024-12-15",
            mode="replay",
            started_at=now - timedelta(days=2),
            finished_at=now - timedelta(days=2),
            counts={"discovered": 40, "high": 15, "borderline": 15, "low": 10},
        )
        session.add(cycle)
        session.flush()

        # Clean existing events for clean demo state
        session.query(FeedbackEvent).filter_by(user_id=user.id).delete()
        session.query(RoutingDecision).filter_by(user_id=user.id).delete()
        session.commit()

        # 20 Positives and 20 Negatives
        for i in range(40):
            pid = f"demo_paper_{i+1:03d}"
            is_pos = (i % 2 == 0)
            if is_pos:
                # Aligned with seed vector
                emb = seed_norm + 0.1 * rng.standard_normal(384).astype(np.float32)
                title = f"Advances in Autonomous LLM Planning and Agent Workflows Part {i+1}"
                route = "HIGH"
                score = 0.82
            else:
                # Orthogonal or opposite
                ortho = rng.standard_normal(384).astype(np.float32)
                ortho -= np.dot(ortho, seed_norm) * seed_norm
                emb = ortho / np.linalg.norm(ortho)
                title = f"Unrelated Classical Quantum Gravity Survey Part {i+1}"
                route = "BORDERLINE" if i < 30 else "LOW"
                score = 0.42

            emb /= np.linalg.norm(emb)
            paper = session.query(Paper).filter_by(id=pid).first()
            if not paper:
                paper = Paper(
                    id=pid,
                    title=title,
                    abstract=f"Abstract for {title}. Investigating agentic autonomy and benchmark evaluation.",
                    submitted_date="2024-12-10",
                    categories=["cs.AI"],
                    embedding=emb.tolist(),
                    embedding_model="all-MiniLM-L6-v2",
                    is_reference=False,
                )
                session.add(paper)

            dec = RoutingDecision(
                user_id=user.id,
                paper_id=pid,
                cycle_id=cycle.id,
                vector_version_id=v0.id,
                cosine_score=score,
                clf_prob=None,
                method="threshold",
                thresholds_used={"t_high": 0.75, "t_low": 0.55},
                route=route,
                status=DecisionStatus.REACTED.value if is_pos else DecisionStatus.ROUTED_BORDERLINE.value,
                reason="Demo simulated routing decision",
                created_at=now - timedelta(days=2),
            )
            session.add(dec)
            session.flush()

            # Add FeedbackEvent (source="user" for positives, source="reviewer" for negatives)
            fe = FeedbackEvent(
                user_id=user.id,
                paper_id=pid,
                source="user" if is_pos else "reviewer",
                label=1 if is_pos else 0,
                decision_id=dec.id,
                created_at=now - timedelta(days=2) + timedelta(minutes=i),
            )
            session.add(fe)

        session.commit()
        logger.info(
            "Demo database snapshot prepared successfully: user='%s', 40 feedback events (20 pos, 20 neg).",
            user_id,
        )
        return user_id

    finally:
        if owns_session:
            session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate oracle feedback for monitoring cycles and prepare demo database snapshot."
    )
    parser.add_argument("--user-id", default="developer", help="User ID to simulate feedback for.")
    parser.add_argument("--cycle-id", type=int, default=None, help="Target cycle ID (default: latest).")
    parser.add_argument("--noise-rate", type=float, default=0.0, help="Probability of label flip (default: 0.0).")
    parser.add_argument(
        "--prepare-demo-snapshot",
        action="store_true",
        help="Prepare complete prerecorded simulated dataset for Phase 14 live demo.",
    )
    args = parser.parse_args()

    if args.prepare_demo_snapshot:
        print("\n" + "=" * 65)
        print("  PREPARING PHASE 14 DEMO DATABASE SNAPSHOT [SIMULATED]")
        print("=" * 65)
        uid = prepare_demo_snapshot()
        print(f"  Demo snapshot created for simulated user: '{uid}'")
        print("  40 effective labels stored (20 positive, 20 negative).")
        print("\n  Ready for instant Phase 14 demo commands:")
        print(f"    python scripts/update_vector.py --user-id {uid}")
        print(f"    python scripts/train_classifier.py --user-id {uid}")
        print("=" * 65 + "\n")
        return 0

    print("\n" + "=" * 65)
    print(f"  APPLYING ORACLE FEEDBACK: USER '{args.user_id}'")
    print("=" * 65)
    counts = apply_oracle_feedback(
        user_id=args.user_id,
        cycle_id=args.cycle_id,
        noise_rate=args.noise_rate,
    )
    print(f"  Reviewer Decisions: {counts['reviewer_events']}")
    print(f"  User Reactions:     {counts['user_events']}")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
