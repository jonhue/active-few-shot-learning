from afsl import ActiveDataLoader, GradientEmebeddings

emb = GradientEmebeddings()
context_loader = ActiveDataLoader(emb, dataset, task, batch_size=5)
data = context_loader.next(model)
model.add_to_context(data)
