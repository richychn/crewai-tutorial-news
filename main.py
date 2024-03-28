from crewai import Crew, Process
from langchain_openai import ChatOpenAI

from agents import AINewsletterAgents
from tasks import AINewsletterTasks
from file_io import save_markdown

from dotenv import load_dotenv
load_dotenv()

OpenAIGPT4 = ChatOpenAI(model='gpt-4')

agents = AINewsletterAgents()
tasks = AINewsletterTasks()

# 1. Set up agents
editor = agents.editor_agent()
news_fetcher = agents.news_fetcher_agent()
news_analyzer = agents.news_analyzer_agent()
newsletter_compiler = agents.newsletter_compiler_agent()

# 2. Set up tasks
# 2.1 Set up callback function save_markdown
fetch_news_task = tasks.fetch_news_task(news_fetcher)
analyze_news_task = tasks.analyze_news_task(news_analyzer, [fetch_news_task])
compile_newsletter_task = tasks.compile_newsletter_task(
    newsletter_compiler, [analyze_news_task], callback_function=save_markdown
)

# 3. Set up tools
# In tools folder

# 4. Create crew
crew = Crew(
    agents=[editor, news_fetcher, news_analyzer, newsletter_compiler],
    tasks=[fetch_news_task, analyze_news_task, compile_newsletter_task],
    process=Process.hierarchical,
    manager_llm=OpenAIGPT4
)

#5. Kickoff crew
results = crew.kickoff()
print("Crew work results: ")
print(results)