# Machine Learning & AI Engineering Guide

## Key Fundamentals
1. **Classical Machine Learning**:
   - Supervised Learning: Linear/Logistic Regression, Decision Trees, Random Forests, Gradient Boosted Trees (XGBoost, LightGBM, CatBoost), SVM.
   - Unsupervised Learning: K-Means, DBSCAN, Hierarchical Clustering, PCA, t-SNE.
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE).
   - Bias-Variance Tradeoff, Regularization (L1 Lasso, L2 Ridge, ElasticNet), Cross-Validation (K-Fold, Stratified K-Fold).
2. **Deep Learning & NLP / RAG**:
   - Neural Networks, Backpropagation, Activation Functions (ReLU, GELU, Sigmoid), Optimization (AdamW, SGD with Momentum).
   - Transformers, Self-Attention Mechanisms, Tokenization (BPE, WordPiece).
   - Retrieval-Augmented Generation (RAG): Document chunking, dense vector embeddings (sentence-transformers), vector stores (ChromaDB, FAISS), re-ranking, and grounded generation with LLMs.
3. **MLOps & Production Deployment**:
   - Model tracking (MLflow, Weights & Biases), model serving (TorchServe, Triton, FastAPI), quantization (INT8, FP16, GGUF), and monitoring data drift.

## Resume Proof Points for ML Roles
- "Built a dense vector RAG search assistant using sentence-transformers and ChromaDB, achieving 94% retrieval precision on internal technical docs."
- "Trained gradient boosted classifier on 500k customer interaction rows to predict subscription churn, improving F1-score from 0.72 to 0.86."
- "Deployed quantized LLM model for automated resume parsing using local inference, reducing cloud API costs by 100% while maintaining 90ms response times."

## Common Technical Interview Questions
- How do you address class imbalance in classification problems?
  - Techniques: Resampling (SMOTE, random oversampling of minority class, downsampling majority), class weighting in loss function, threshold tuning, and using Precision-Recall AUC rather than ROC-AUC.
- Explain the difference between Cosine Similarity and Dot Product in embedding retrieval.
  - Dot product is the magnitude-weighted product; cosine similarity is dot product normalized by vector lengths. When embeddings are L2-normalized ($||v|| = 1$), dot product equals cosine similarity.
