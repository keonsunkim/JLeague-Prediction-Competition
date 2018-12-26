import sys

from MLJleague.Scraper.j1j2scraper import main_scraper


scripts_dict = {'get_extra_data': main_scraper}

if __name__ == '__main__':
    try:
        scripts_dict[sys.argv[1]]()
    except KeyError:
        msg = ('The main module only accepts the following arguments',
               '"get_extra_data", "preprocess", ... add more later')
        raise KeyError(msg)
