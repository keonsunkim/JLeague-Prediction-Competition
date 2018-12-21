"""
Utility functions for our scraper
"""
import time
from random import randrange

import requests
from bs4 import BeautifulSoup


def random_waiter(min, max):
    """
    waits within a random min, max second (parameters are miliseconds)
    """
    time.sleep(randrange(min, max) * 0.1)


def connection_checker(response):
    """
    checks if the requests recieved a success http status code
    """
    success_http_status_code = (200, 201, 202)

    if response.status_code not in success_http_status_code:
        msg = 'unable to connect to "data.j-league.or.jp"'
        + 'tyring with different ip and user-agent'
        raise ConnectionError(msg)


def check_and_find_html_elements(html, *args, **kwargs):
    """
    there can be cases which the html response may give out
    200 status code, but lacks the information we aim to scrape.

    ex) "https://data.j-league.or.jp/SFMS01/search?
        competition_years=2017&competition_frame_ids=13"

    In the case above, the get method parameter did not comply
    properly to the website's get url logic. Thus recieved a
    webpage with no search results.

    We intend to catch problems like this.
    """
    nested = kwargs.pop('nested', False)

    soup = BeautifulSoup(html, 'html.parser')
    page_soup = BeautifulSoup(html, 'html.parser')

    for element in args:
        html_tag = element[0]
        html_attribute = element[-1] if element[-1] else dict()

        if nested:
            soup = soup.find(html_tag, attrs=html_attribute)
        else:
            soup = page_soup.find(html_tag, attrs=html_attribute)

        if soup is None:
            msg = 'The html does not have {} element!'.format(element)
            raise ValueError(msg)

    if not nested:
        html_tag = args[-1][0]
        html_attribute = args[-1][-1] if args[-1][-1] else dict()
        soup = page_soup.find(args[-1][0], attrs=args[-1][-1])

    return soup


# class RequestsMaintainClass(objcet):
#     """
#     Using class object to constantly check whether connection is
#     properly set between the website and user.

#     This allows instant checking and automatic retrying with the
#     website.
#     """

#     def __init__(self, session, status, **kwargs):
#         self.session = session
#         self.status = status
#         self.availible_vpn = None
#         self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
#                            AppleWebKit/537.36 (KHTML, like Gecko) \
#                            Chrome/70.0.3538.77 Safari/537.36'
#         """
#         I use chrome 71 to inspect the html, therefore I matched
#         the user_agent to try my best to guarantee that my scraper
#         recieves the same html information from the website
#         """

#         for options in ('cut_connection_if_vpn_depleted', 'max_trial'):
#             self.o
#         self.cut_connection_if_vpn_depleted = cut_connection_if_vpn_depleted

#     def reset_connection(self):
#         if self.cut_connection_if_vpn_depleted:
#             if availible_vpn is None:
#                 msg = 'unable to connect to "data.j-league.or.jp"'
#                     + 'tyring with different ip and user-agent'
#                 raise ConnectionError(msg)

#     def connection_checker(self, response):
#         """
#         checks if the requests recieved a success http status code
#         """
#         success_http_status_code = (200, 201, 202)

#         if response.status not in success_http_status_code:
#         msg = 'unable to connect to "data.j-league.or.jp"'
#             + 'tyring with different ip and user-agent'
#             raise ConnectionError(msg)
