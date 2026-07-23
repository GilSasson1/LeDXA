"""Repository-relative paths used by the Figure 6 generator."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TABLES_DIR = os.path.join(REPO_ROOT, 'tables')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')


def out_table(name: str) -> str:
    os.makedirs(TABLES_DIR, exist_ok=True)
    return os.path.join(TABLES_DIR, name)


def out_figure(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)
