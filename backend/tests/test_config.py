from resume_screener.config import sqlalchemy_url


def test_neon_urls_get_the_psycopg_driver():
    assert sqlalchemy_url("postgresql://u:p@h/db?sslmode=require") == "postgresql+psycopg://u:p@h/db?sslmode=require"
    assert sqlalchemy_url("postgres://u:p@h/db").startswith("postgresql+psycopg://")
    assert sqlalchemy_url("sqlite:///./local.db") == "sqlite:///./local.db"
