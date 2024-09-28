from activeft import ActiveDataLoader

context_loader = ActiveDataLoader.initialize(dataset, target, batch_size=5)
context = dataset[context_loader.next(model)]
model.add_to_context(context)
