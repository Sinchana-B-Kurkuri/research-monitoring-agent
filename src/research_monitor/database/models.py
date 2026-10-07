from sqlalchemy import Column, String, Boolean, JSON, Text
from sqlalchemy.orm import declarative_base
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
