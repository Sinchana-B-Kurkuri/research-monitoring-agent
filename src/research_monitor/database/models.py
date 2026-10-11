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

    def __repr__(self):
        return (
            f"<RoutingDecision(user_id='{self.user_id}', paper_id='{self.paper_id}', "
            f"route='{self.route}', status='{self.status}', score={self.cosine_score:.4f})>"
        )
