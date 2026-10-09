from datasets import load_dataset

for config in ["pupa_tnb", "pupa_new"]:
    data = load_dataset("Columbia-NLP/PUPA", config)
    print(config, data)
    for split in data:
        data[split].to_csv(f"data/{config}_{split}.csv")
        print("Columns:", data[split].column_names)
        print("First row:", data[split][0], "\n")