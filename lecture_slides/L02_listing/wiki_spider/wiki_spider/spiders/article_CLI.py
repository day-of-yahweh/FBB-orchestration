import scrapy
class ArticleCLISpider(scrapy.Spider):
    name = 'article_CLI'
    def __init__(self, urls=None, *args, **kwargs):
        super(ArticleCLISpider, self).__init__(*args, **kwargs)
        if urls:
            self.start_urls = [url.strip() for url in urls.split(',')]
        else:
            self.start_urls = []

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        url = response.url
        title = response.xpath('string(//h1[@id="firstHeading"])').get().strip()
        print(f'URL is: {url}')
        print(f'Title is: {title}')
