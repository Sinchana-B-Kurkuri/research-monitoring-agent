from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Boolean,
    JSON,
    Text,
    Integer,
    Float,
    DateTime,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, synonym, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Paper(Base):
    __tablename__ = 'papers'

    id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    submitted_date = Column(String, nullable=False)
    categories = Column(JSON, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    embedding_model = Column(String, nullable=False)
    is_reference = Column(Boolean, nullable=False)

    def __repr__(self):
        return f"<Paper(id='{self.id}', title='{self.title[:30]}...')>"


class ValidationLabel(Base):
    """
    Validation label for threshold calibration and offline evaluation (Roadmap Section 6).

    Persists ground truth ratings ('relevant', 'somewhat', 'not'/'not_relevant'),
    alongside sampling stratum (S1-S5), sampling weight, split, and timestamps.
    Unique constraint on (user_id, paper_id) prevents accidental duplicate entries.
    """
    __tablename__ = 'validation_labels'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, default="developer", nullable=False)
    paper_id = Column(String, nullable=False, index=True)
    label = Column(String, nullable=False)  # 'relevant', 'somewhat', 'not' (or 'not_relevant')
    sampling_stratum = Column(String, nullable=False)  # 'S1', 'S2', 'S3', 'S4', 'S5'
    sampling_weight = Column(Float, nullable=False)
    split = Column(String, nullable=False)  # 'dev', 'test', 'calibration', 'pilot_test'
    notes = Column(Text, nullable=True)
    labelled_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint('user_id', 'paper_id', name='uq_validation_labels_user_paper'),
    )

    # Backwards-compatible synonyms for Phase 4 code
    stratum = synonym("sampling_stratum")
    created_at = synonym("labelled_at")

    def __repr__(self):
        return (
            f"<ValidationLabel(user_id='{self.user_id}', paper_id='{self.paper_id}', "
            f"label='{self.label}', stratum='{self.sampling_stratum}', split='{self.split}')>"
        )


class User(Base):
    """
    User entity representing pilot or simulated users (Roadmap Section 6).
    """
    __tablename__ = 'users'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    access_token = Column(String, nullable=True)
    is_simulated = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    profiles = relationship("InterestProfile", back_populates="user", cascade="all, delete-orphan")
    vector_versions = relationship("InterestVectorVersion", back_populates="user", cascade="all, delete-orphan")
    cycles = relationship("Cycle", back_populates="user", cascade="all, delete-orphan")
    routing_decisions = relationship("RoutingDecision", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id='{self.id}', name='{self.name}', is_simulated={self.is_simulated})>"


class InterestProfile(Base):
    """
    Seed profile text and vector for a user (Roadmap Section 6).
    """
    __tablename__ = 'interest_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    seed_text = Column(Text, nullable=False)
    seed_embedding = Column(Vector(384), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="profiles")

    def __repr__(self):
        return f"<InterestProfile(id={self.id}, user_id='{self.user_id}')>"


class InterestVectorVersion(Base):
    """
    Preference memory: routing vector version with active percentile thresholds (Roadmap Section 6 & 7.2).
    """
    __tablename__ = 'interest_vector_versions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    vector = Column(Vector(384), nullable=False)
    method = Column(String, default="seed", nullable=False)  # 'seed' or 'rocchio'
    n_effective_labels = Column(Integer, default=0, nullable=False)
    built_from_max_event_id = Column(Integer, nullable=True)
    p_high_level = Column(Float, nullable=False)
    p_low_level = Column(Float, nullable=False)
    t_high = Column(Float, nullable=False)
    t_low = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'version', name='uq_user_vector_version'),
    )

    user = relationship("User", back_populates="vector_versions")

    def __repr__(self):
        return (
            f"<InterestVectorVersion(id={self.id}, user_id='{self.user_id}', "
            f"version={self.version}, is_active={self.is_active}, t_high={self.t_high:.4f}, t_low={self.t_low:.4f})>"
        )


class Cycle(Base):
    """
    Monitoring run record supporting replay or live execution modes (Roadmap Section 6 & 8.6).
    """
    __tablename__ = 'cycles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    as_of_date = Column(String, nullable=False)
    mode = Column(String, nullable=False)  # 'replay' or 'live'
    started_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    finished_at = Column(DateTime(timezone=True), nullable=True)
    counts = Column(JSON, nullable=True)
    config_snapshot = Column(JSON, nullable=True)

    user = relationship("User", back_populates="cycles")
    routing_decisions = relationship("RoutingDecision", back_populates="cycle")

    def __repr__(self):
        return (
            f"<Cycle(id={self.id}, user_id='{self.user_id}', mode='{self.mode}', as_of_date='{self.as_of_date}')>"
        )


class RoutingDecision(Base):
    """
    Core auditable routing decision per (user, paper) including LOW papers (Roadmap Section 6 & 9.1).
    """
    __tablename__ = 'routing_decisions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String, ForeignKey('papers.id'), nullable=False, index=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    cycle_id = Column(Integer, ForeignKey('cycles.id'), nullable=True, index=True)
    cosine_score = Column(Float, nullable=False)
    vector_version_id = Column(Integer, ForeignKey('interest_vector_versions.id'), nullable=False, index=True)
    clf_prob = Column(Float, nullable=True)  # null at cold start
    method = Column(String, default="threshold", nullable=False)  # 'threshold' or 'classifier'
    model_version_id = Column(Integer, nullable=True)  # null at cold start
    thresholds_used = Column(JSON, nullable=False)
    route = Column(String, nullable=False)   # 'HIGH', 'BORDERLINE', 'LOW'
    status = Column(String, nullable=False)  # 'ROUTED_HIGH', 'ROUTED_BORDERLINE', 'DISCARDED'
    reason = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint('user_id', 'paper_id', name='uq_user_paper_routing_decision'),
    )

    user = relationship("User", back_populates="routing_decisions")
    paper = relationship("Paper")
    cycle = relationship("Cycle", back_populates="routing_decisions")
    vector_version = relationship("InterestVectorVersion")
    review_tasks = relationship("ReviewTask", back_populates="routing_decision", cascade="all, delete-orphan")
    digest_items = relationship("DigestItem", back_populates="routing_decision", cascade="all, delete-orphan")
    feedback_events = relationship("FeedbackEvent", back_populates="routing_decision", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            f"<RoutingDecision(user_id='{self.user_id}', paper_id='{self.paper_id}', "
            f"route='{self.route}', status='{self.status}', score={self.cosine_score:.4f})>"
        )


class ReviewTask(Base):
    """
    Human review queue item for borderline or audit papers (Roadmap Section 6 & 7.6).
    """
    __tablename__ = 'review_tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, ForeignKey('routing_decisions.id'), nullable=False, index=True)
    kind = Column(String, default="borderline", nullable=False)  # 'borderline' or 'audit'
    queued_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    shown_at = Column(DateTime(timezone=True), nullable=True)
    decision = Column(String, nullable=True)  # 'accept', 'reject', 'skip', 'expired'
    decided_at = Column(DateTime(timezone=True), nullable=True)
    expired_at = Column(DateTime(timezone=True), nullable=True)

    routing_decision = relationship("RoutingDecision", back_populates="review_tasks")

    def __repr__(self):
        return f"<ReviewTask(id={self.id}, decision_id={self.decision_id}, kind='{self.kind}', decision='{self.decision}')>"


class Digest(Base):
    """
    Delivery digest for a user per cycle (Roadmap Section 6 & 7.6).
    """
    __tablename__ = 'digests'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    cycle_id = Column(Integer, ForeignKey('cycles.id'), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint('user_id', 'cycle_id', name='uq_user_cycle_digest'),
    )

    user = relationship("User")
    cycle = relationship("Cycle")
    items = relationship(
        "DigestItem",
        back_populates="digest",
        cascade="all, delete-orphan",
        order_by="DigestItem.position",
    )

    def __repr__(self):
        return f"<Digest(id={self.id}, user_id='{self.user_id}', cycle_id={self.cycle_id})>"


class DigestItem(Base):
    """
    Individual delivered paper within a digest (Roadmap Section 6 & 7.6).
    """
    __tablename__ = 'digest_items'

    id = Column(Integer, primary_key=True, autoincrement=True)
    digest_id = Column(Integer, ForeignKey('digests.id'), nullable=False, index=True)
    decision_id = Column(Integer, ForeignKey('routing_decisions.id'), nullable=False, index=True)
    position = Column(Integer, nullable=False)  # 1-based ordering
    via = Column(String, nullable=False)        # 'auto' or 'review'
    shown_at = Column(DateTime(timezone=True), nullable=True)

    digest = relationship("Digest", back_populates="items")
    routing_decision = relationship("RoutingDecision", back_populates="digest_items")

    def __repr__(self):
        return f"<DigestItem(id={self.id}, digest_id={self.digest_id}, decision_id={self.decision_id}, position={self.position}, via='{self.via}')>"


class FeedbackEvent(Base):
    """
    User and reviewer feedback signals (append-only) (Roadmap Section 6 & 7.3).
    """
    __tablename__ = 'feedback_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    paper_id = Column(String, ForeignKey('papers.id'), nullable=False, index=True)
    decision_id = Column(Integer, ForeignKey('routing_decisions.id'), nullable=True, index=True)
    source = Column(String, nullable=False)   # 'user' or 'reviewer'
    label = Column(Integer, nullable=False)    # 1 or 0
    is_audit = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User")
    paper = relationship("Paper")
    routing_decision = relationship("RoutingDecision", back_populates="feedback_events")

    def __repr__(self):
        return f"<FeedbackEvent(id={self.id}, user_id='{self.user_id}', paper_id='{self.paper_id}', source='{self.source}', label={self.label})>"


class TrainingLabel(Base):
    """
    ORM representation of the training_labels database view (Roadmap Section 6 & 7.3).
    Provides one effective label per (user_id, paper_id) where latest user like/dislike
    takes precedence over reviewer decisions.
    """
    __tablename__ = 'training_labels'

    event_id = Column(Integer, primary_key=True)
    user_id = Column(String, primary_key=True)
    paper_id = Column(String, primary_key=True)
    label = Column(Integer, nullable=False)
    origin = Column(String, nullable=False)

    def __repr__(self):
        return (
            f"<TrainingLabel(user_id='{self.user_id}', paper_id='{self.paper_id}', "
            f"label={self.label}, origin='{self.origin}', event_id={self.event_id})>"
        )


def create_training_labels_view(bind_or_engine) -> None:
    """
    Creates or replaces the training_labels database view in PostgreSQL or SQLite.
    Rule (Section 7.3): Most recent user like/dislike takes precedence over reviewer decision.
    Skips and expired tasks create no feedback events and thus produce no labels.
    """
    from sqlalchemy import text
    dialect_name = getattr(bind_or_engine, "dialect", None)
    dialect_str = dialect_name.name if dialect_name else "postgresql"

    view_sql = """
    WITH ranked_events AS (
        SELECT
            fe.id AS event_id,
            fe.user_id,
            fe.paper_id,
            fe.label,
            fe.source AS origin,
            fe.created_at,
            ROW_NUMBER() OVER (
                PARTITION BY fe.user_id, fe.paper_id
                ORDER BY
                    CASE WHEN fe.source = 'user' THEN 1 ELSE 2 END,
                    fe.created_at DESC,
                    fe.id DESC
            ) AS rank
        FROM feedback_events fe
    )
    SELECT
        event_id,
        user_id,
        paper_id,
        label,
        origin
    FROM ranked_events
    WHERE rank = 1;
    """

    if hasattr(bind_or_engine, "begin"):
        cm = bind_or_engine.begin()
    else:
        # already a connection or session
        from contextlib import nullcontext
        cm = nullcontext(bind_or_engine)

    with cm as conn:
        if dialect_str == "postgresql":
            conn.execute(text("DROP VIEW IF EXISTS training_labels CASCADE;"))
            conn.execute(text(f"CREATE OR REPLACE VIEW training_labels AS {view_sql}"))
        else:
            conn.execute(text("DROP TABLE IF EXISTS training_labels;"))
            conn.execute(text("DROP VIEW IF EXISTS training_labels;"))
            conn.execute(text(f"CREATE VIEW training_labels AS {view_sql}"))
