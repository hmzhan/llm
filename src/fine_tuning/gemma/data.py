

ds = Dataset.from_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")
ds = ds.select(torch.arange(100))  # We only use the first 100 data for demo purpose

