from crewai_tools import BaseTool, ScrapeWebsiteTool, SerperDevTool

from datetime import date

class TodayTool(BaseTool):
    name: str ="Today Tool"
    description: str = ("Gets today's date.")
    
    def _run(self) -> str:
        return (str(date.today()))

def today_tool():
    return TodayTool()

def search_tool():
    return SerperDevTool()

def scrape_tool():
    return ScrapeWebsiteTool()
