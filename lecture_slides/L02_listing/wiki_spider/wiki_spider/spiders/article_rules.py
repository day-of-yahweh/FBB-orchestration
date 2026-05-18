from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class WikiSpider(CrawlSpider):
    name = 'wiki'
    allowed_domains = ['en.wikipedia.org']
    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url] if start_url else [
            'https://en.wikipedia.org/wiki/Benevolent_dictator_for_life'
        ]
        self.rules = (
            Rule(
                LinkExtractor(
                    allow=r'^https?://en\.wikipedia\.org/wiki/[A-Za-z0-9\-\'\.\,]+$',
                    deny=r'/\.(?:jpg|jpeg|png|gif|pdf)$',
                ),
                callback='parse_article',
                follow=True,
            ),
        )
        self._compile_rules()

    def parse_article(self, response):
        # Verify we're on an actual article page
        if not response.css('h1#firstHeading'):
            self.logger.warning(f"Not an article page: {response.url}")
            return

        title = response.xpath('string(//h1[@id="firstHeading"])').get()
        last_updated = response.css('#footer-info-lastmod::text').re_first(
            r'edited on ([^<]+)'
        )
        yield {
            'url': response.url,
            'title': title,
            'last_updated': last_updated,
        }
