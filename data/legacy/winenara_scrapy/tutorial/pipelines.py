# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import scrapy
from scrapy.pipelines.images import ImagesPipeline
from urllib.parse import urlparse
import os

class TutorialPipeline:
    def process_item(self, item, spider):
        return item

class CustomImagesPipeline(ImagesPipeline):
    def file_path(self, request, response=None, info=None, *, item=None):
        path = urlparse(request.url).path
        path_parts = path.split('/')
        filename = os.path.join(*path_parts[-3:])
        return filename

    def get_media_requests(self, item, info):
        if 'img_url' in item:
            for image_url in item['img_url']:
                yield scrapy.Request(image_url)
