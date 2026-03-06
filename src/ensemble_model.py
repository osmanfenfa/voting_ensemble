from __future__ import annotations
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def build_voting_classifier(random_state: int = 42) -> VotingClassifier:
    decision_tree = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
    )
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

    return VotingClassifier(
        estimators=[("decision_tree", decision_tree), ("knn", knn)],
        voting="soft",
    )