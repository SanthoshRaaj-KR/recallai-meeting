from abc import ABC, abstractmethod

class DocumentFetcher(ABC):
    @abstractmethod
    def fetch_page_html(self, page_id: str) -> str:
        """Fetches the HTML or Atlassian Document Format payload of a document."""
        pass

class DocumentPusher(ABC):
    @abstractmethod
    def push_update(self, page_id: str, content: str) -> bool:
        """Pushes an update to a document."""
        pass
