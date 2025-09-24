from llmjj.prompt_builder import PromptBuilder

def main():
    builder = PromptBuilder.from_files("conf/base.yaml", "conf/query_doc_helpfulness.yaml")
    prompt = builder.render()
    print(prompt)


if __name__ == "__main__":
    main()
