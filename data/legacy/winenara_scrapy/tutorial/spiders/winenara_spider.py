import scrapy
from scrapy_splash import SplashRequest

# 아래는 와인 나라에서 크롤링을 해온다.
class WinenaraSpider(scrapy.Spider):
    name = "winenara"

    start_urls = ["https://www.winenara.com/shop/product/product_lists?sh_category1_cd=10000&sh_category2_cd=10100&sh_category3_cd=&sh_order_by=all&sh_sort_order_by=&sh_filter_code=&sh_rcd="]

    def parse(self, response):
        detail_page_links = response.css(".table_box::attr(href)")
        yield from response.follow_all(detail_page_links, self.parse_winenara)

        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

    def parse_vivino_page(self, response):
        # parse the vivino page here
        def extract_with_css(query):
            return response.css(query).get(default="").strip()
        
        def extract_features():            
            details = {}
            details_sections = response.css('.tasteStructure__tasteCharacteristic--jLtsE').getall()
            for section in details_sections:
                current_key = section.css('.tasteStructure__property--CLNl_::text').get().lower().strip()
                style = response.css('span.indicatorBar__progress--3aXLX::attr(style)').get()
                left = [s.strip() for s in style.split(';') if 'left' in s][0]
                left_value = left.split(":")[1].strip()
                current_value = str(float(left_value)/100)
                if current_key and current_value:
                    details[current_key] = current_value.lower()
            return details
        
        def extract_info():
            details = {}
            details_sections = response.css('.breadCrumbs__link--1TY6b span').getall()
            for section in details_sections:
                current_key = section.css('a::attr(data-cy)').get().lower().strip()
                current_value = section.css('a::text').get().lower().strip()
                if current_key and current_value:
                    details[current_key] = current_value.lower()
            return details

        data_dict = response.meta['data_dict']
        
        data_dict['vivino']['rating'] = extract_with_css('.vivinoRating_averageValue__uDdPM::text')
        data_dict['vivino']['rating_num'] = extract_with_css('.vivinoRating_caption__xL84P::text')
        data_dict['vivino']['price'] = extract_with_css('.purchaseAvailability__currentPrice--3mO4u::text')
        data_dict['vivino']['features'] = extract_features()
        data_dict['vivino']['food_pairing'] = response.css('.foodPairing__foodImage--2OYHg::attr(aria-label)').getall()
        data_dict['vivino']['info'] = extract_info()

        yield data_dict

    def parse_winenara(self, response):
        data_dict = {}
        def extract_with_css(query):
            return response.css(query).get(default="").strip()
        
        def extract_features():            
            details = {}
            details_sections = response.css('dl.details')
            for section in details_sections:
                current_key = section.css('dt::text').get().lower().strip()
                current_value = section.css('dd span.label[style*="background"]::text').get()
                if current_key and current_value:
                    details[current_key] = current_value.lower()
            return details

        def extract_tag():
            return response.css(".cate_label .label::text").getall()
        
        def extract_img_url(query):
            img_src = response.css(query).get(default="")
            if img_src:
                img_url = response.urljoin(img_src)  # Assuming img_src is a relative url
                return img_url

        


        data_dict['url'] = response.url
        data_dict['price'] = extract_with_css("p ins::text")
        data_dict['name'] = extract_with_css(".prd_name::text")
        data_dict['en_name'] = extract_with_css(".prd_en_name::text")
        data_dict['img_url'] = [extract_img_url(".lozad::attr(data-src)")]
        data_dict['features'] = extract_features()
        data_dict['feature_img_url'] = [extract_img_url(".tab_con > p > img::attr(src)")]
        data_dict['tag'] = extract_tag()
        data_dict['rating'] = extract_with_css(".info strong::text")
        data_dict['vivino_link'] = extract_with_css("div.box > a::attr(href)")  
        # data_dict['vivino'] = {}
        if data_dict['vivino_link'] == '':
            yield None
        else:
            yield data_dict