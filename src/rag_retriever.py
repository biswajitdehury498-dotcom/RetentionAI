
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetentionAIRetriever:
    """Lightweight local retriever for RetentionAI knowledge."""

    REQUIRED_COLUMNS = ["doc_id", "topic", "content"]

    def __init__(self, knowledge_csv_path=None):
        if knowledge_csv_path is None:
            project_root = Path(__file__).resolve().parents[1]
            knowledge_csv_path = (
                project_root
                / "data"
                / "processed"
                / "retentionai_rag_knowledge_chunks.csv"
            )

        self.knowledge_csv_path = Path(knowledge_csv_path)

        if not self.knowledge_csv_path.exists():
            raise FileNotFoundError(
                f"Knowledge file not found: {self.knowledge_csv_path}"
            )

        self.documents = pd.read_csv(self.knowledge_csv_path)

        missing = set(self.REQUIRED_COLUMNS) - set(
            self.documents.columns
        )

        if missing:
            raise ValueError(
                f"Knowledge file is missing columns: {sorted(missing)}"
            )

        if self.documents.empty:
            raise ValueError("The knowledge base is empty.")

        self.documents["content"] = (
            self.documents["content"].fillna("").astype(str)
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.document_vectors = self.vectorizer.fit_transform(
            self.documents["content"]
        )

    @staticmethod
    def expand_query(query):
        """Add domain terms to improve lexical retrieval."""

        text = query.lower()
        extra_terms = []

        if (
            ("high-value" in text or "high value" in text)
            and any(term in text for term in ["risk", "monetary"])
        ):
            extra_terms.extend([
                "High-Value High-Risk Customers",
                "historical monetary value",
                "average_order_value",
                "customer_unique_id",
                "retention_risk",
                "repeat_purchase_probability",
                "frequency recency_days"
            ])

        elif "campaign" in text or "prioritize" in text:
            extra_terms.extend([
                "Campaign Priorities",
                "campaign priority levels customer counts",
                "recommended_action",
                "campaign_priority",
                "customer_count",
                "customer_percentage",
                "average_priority_score",
                "average_retention_risk",
                "average_monetary",
                "Service Recovery",
                "High-Value Win-Back",
                "Re-Engagement Campaign",
                "Personalized Retention Offer",
                "Nurture Campaign",
                "Loyalty Engagement"
            ])

        elif (
            "critical-risk" in text
            or "critical risk" in text
            or (
                ("poor review" in text or "bad review" in text)
                and any(
                    term in text
                    for term in ["canceled", "cancelled", "unavailable"]
                )
            )
        ):
            extra_terms.extend([
                "Critical-Risk Customers",
                "average_review_score",
                "canceled_orders",
                "unavailable_orders",
                "delivery_success_rate",
                "critical_risk_customer_count"
            ])

        elif any(term in text for term in [
            "highest risk",
            "highest-risk",
            "highest retention risk",
            "warning signals"
        ]):
            extra_terms.extend([
                "Highest-Risk Customer Examples",
                "customer_unique_id",
                "retention_risk",
                "repeat_purchase_probability",
                "frequency",
                "recency_days",
                "average_review_score",
                "delivery_success_rate"
            ])

        elif any(term in text for term in [
            "risk segment",
            "risk distribution",
            "risk levels"
        ]):
            extra_terms.extend([
                "Customer Risk Segments",
                "risk_level",
                "customer_count",
                "customer_percentage",
                "average_retention_risk",
                "average_repeat_probability"
            ])

        return query + " " + " ".join(extra_terms)

    def retrieve(self, query, top_k=3, min_similarity=0.08):
        """Return the most relevant knowledge documents."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if not 0 <= min_similarity <= 1:
            raise ValueError(
                "min_similarity must be between 0 and 1."
            )

        expanded_query = self.expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])

        scores = cosine_similarity(
            query_vector,
            self.document_vectors
        ).flatten()

        ranked_indices = scores.argsort()[::-1]

        selected = [
            idx for idx in ranked_indices
            if scores[idx] >= min_similarity
        ][:top_k]

        columns = self.REQUIRED_COLUMNS + ["similarity_score"]

        if not selected:
            return pd.DataFrame(columns=columns)

        results = self.documents.iloc[selected][
            self.REQUIRED_COLUMNS
        ].copy()

        results["similarity_score"] = [
            float(scores[idx]) for idx in selected
        ]

        return results.reset_index(drop=True)
