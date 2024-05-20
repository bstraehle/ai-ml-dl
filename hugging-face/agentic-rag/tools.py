from crewai_tools import ScrapeWebsiteTool, SerperDevTool

def search_tool():
    return SerperDevTool()

def scrape_tool():
    return ScrapeWebsiteTool()
