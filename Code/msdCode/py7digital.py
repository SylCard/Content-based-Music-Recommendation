#!/usr/bin/env python
"""A python interface to 7Digital web service (non-premium services)"""
import os
import urllib2, urllib
import re
import urlparse
from xml.dom import minidom
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

__name__ = 'py7digital'
__doc__ = 'A python interface to 7Digital web service'
__author__ = 'Oscar Celma, Pau Capella'
__version__ = '0.0.1'
__license__ = 'GPL'
__maintainer__ = 'Oscar Celma'
__email__ = 'ocelma@bmat.com'
__status__ = 'Beta'

API_VERSION = '1.2'
HOST_NAME = 'api.7digital.com/' + API_VERSION
OAUTHKEY = '7d39j569kyht' #TODO Put your oauth key here
COUNTRY = 'GB'  # ISO Country

__cache_dir = './cache' # Set cache directory
__cache_enabled = False  # Enable cache? if set to True, make sure that __cache_dir exists! (e.g. $ mkdir ./cache)

class ServiceException(Exception):
    """Exception related to the web service."""

    def __init__(self, type, message):
        self._type = type
        self._message = message

    def __str__(self):
        return self._type + ': ' + self._message

    def get_message(self):
        return self._message

    def get_type(self):
        return self._type

class _Request(object):
    """Representing an abstract web service operation."""

    def __init__(self, method_name, params):
        self.params = params
        self.method = method_name

    def _download_response(self):
        """Returns a response"""
        data = []
        for name in self.params.keys():
            data.append('='.join((name, urllib.quote_plus(self.params[name].replace('&amp;', '&').encode('utf8')))))
        data = '&'.join(data)

        url = HOST_NAME
        parsed_url = urlparse.urlparse(url)
        if not parsed_url.scheme:
            url = "http://" + url
        url += self.method + '?oauth_consumer_key=' + OAUTHKEY + '&'
        if COUNTRY:
            url += 'country=' + COUNTRY + '&'
        url += data
        #print url

        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        return response.read()

    def execute(self, cacheable=False):
        try:
            if is_caching_enabled() and cacheable:
                response = self._get_cached_response()
            else:
                response = self._download_response()
            return minidom.parseString(response)
        except urllib2.HTTPError, e:
            raise self._get_error(e.fp.read())

    def _get_cache_key(self):
        """Cache key"""
        keys = self.params.keys()[:]
        keys.sort()
        string = self.method
        for name in keys:
            string += name
            string += self.params[name]
        return get_md5(string)

    def _is_cached(self):
        """Returns True if the request is available in the cache."""
        return os.path.exists(os.path.join(_get_cache_dir(), self._get_cache_key()))

    def _get_cached_response(self):
        """Returns a file object of the cached response."""
        if not self._is_cached():
            response = self._download_response()
            response_file = open(os.path.join(_get_cache_dir(), self._get_cache_key()), "w")
            response_file.write(response)
            response_file.close()
        return open(os.path.join(_get_cache_dir(), self._get_cache_key()), "r").read()

    def _get_error(self, text):
        return ServiceException('Error', text)
        raise


class _BaseObject(object):
    """An abstract webservices object."""

    def __init__(self, method):
        self._method = method
        self._xml = None

    def _request(self, method_name , cacheable = False, params = None):
        if not params:
            params = self._get_params()
        return _Request(method_name, params).execute(cacheable)

    def _get_params(self):
        return dict()


class Artist(_BaseObject):
    """ A 7digital artist """

    def __init__(self, id):
        _BaseObject.__init__(self, '/artist/')

        self.name = None
        self.id = id
        self._url = None
        self._image = None
        self._albums = None
        self._top_tracks = None
        self._tags = None
        self._recommended_albums = None

    def __repr__(self):
       return self.get_name().encode('utf8')

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __ne__(self, other):
        return self.get_id() != other.get_id()

    def get_id(self):
        """ Returns the 7digital artist id """
        return self.id

    def get_name(self):
        """ Returns the name of the artist """
        if self.name is None:
            self.name = ''
            try:
                if not self._xml : self._xml = self._request(self._method, + 'details', True, {'artistid': self.id})
                self.name = _extract(self._xml, 'artist', 1) or ''
            except:
                return self.name
        return self.name

    def set_name(self, name) :
        self.name = name

    def get_image(self):
        """ Returns the image url of an artist """
        if self._image is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'artistid': self.id})
            self._image = _extract(self._xml, 'image')
        return self._image

    def set_image(self, image) :
        self._image = image

    def get_url(self):
        """ Returns the url of an artist """
        if self._url is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'artistid': self.id})
            self._url = _extract(self._xml, 'url')
        return self._url

    def set_url(self, url) :
        self._url = url

    def get_tags(self, pageSize=10):
        """ Returns the tags of an artist """
        if self._tags is None:
            self._tags = []
            xml = self._request(self._method + 'tags', True, {'artistid': self.id})
            for node in xml.getElementsByTagName('tag') :
                self._tags.append(_get_tag(node))
        if self._tags:
            self._tags.sort()
        return self._tags[:pageSize]

    def get_albums(self, pageSize=10):
        """ Returns the albums of an artist """
        if self._albums is not None: return self._albums

        results = []
        xml = self._request(self._method + 'releases', True, {'artistid': self.id, 'pageSize': str(pageSize)})
        for node in xml.getElementsByTagName('release'):
            album = _get_album(node, self)
            results.append(album)
        self._albums = results
        return self._albums

    def get_recommended_albums(self, pageSize=10):
        """ Returns a list of recommended albums based on the seed artist """
        if self._recommended_albums is not None: return self._recommended_albums

        results = []
        xml = self._request('/release/recommend', True, {'artistid': self.id, 'pageSize': str(pageSize)}) # TODO if country is set gives different results
        for node in xml.getElementsByTagName('release'):
            results.append(_get_album(node, _get_artist(node.getElementsByTagName('artist')[0])))
        self._recommended_albums = results
        return self._recommended_albums

    def get_top_tracks(self, pageSize=10):
        """ Returns the top tracks of an artist """
        if self._top_tracks is not None: return self._top_tracks

        results = []
        xml = self._request(self._method + 'toptracks', True, {'artistid': self.id, 'pageSize': str(pageSize)})
        for node in xml.getElementsByTagName('track'):
            results.append(_get_track(node, None, self))
        self._top_tracks = results
        return self._top_tracks

class Album(_BaseObject):
    """ A 7digital album """
    def __init__(self, id=HOST_NAME):
        _BaseObject.__init__(self, '/release/')
        self.id = id
        self.artist = None
        self.title = None

        self._url = None
        self._type = None
        self._barcode = None
        self._year = None
        self._image = None
        self._label = None
        self._tags = None
        self._tracks = None
        self._similar = None
        self._release_date = None
        self._added_date = None

    def __repr__(self):
        if self.get_artist():
            return self.get_artist().get_name().encode('utf8') + ' - ' + self.get_title().encode('utf8')
        return self.get_title().encode('utf8')

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __ne__(self, other):
        return self.get_id() != other.get_id()

    def get_id(self):
        """ Returns the 7digital album id """
        return self.id

    def get_title(self):
        """ Returns the name of the album """
        if self.title is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self.title = _extract(self._xml, 'release', 1)
        return self.title

    def set_title(self, title):
        if title is None:
            title = ''
        self.title = title

    def get_type(self):
        """ Returns the type (CD, DVD, etc.) of the album """
        if self._type is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self._type = _extract(self._xml, 'type')
        return self._type

    def set_type(self, type):
        self._type = type

    def get_year(self):
        """ Returns the year of the album """
        if self._year is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self._year = _extract(self._xml, 'year')
        return self._year

    def set_year(self, year):
        self._year = year

    def get_barcode(self):
        """ Returns the barcode of the album """
        if self._barcode is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self.barcode = _extract(self._xml, 'barcode')
        return self._barcode

    def set_barcode(self, barcode):
        self._barcode = barcode

    def get_url(self):
        """ Returns the url of the album """
        if self._url is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self._url = _extract(self._xml, 'url')
        return self._url

    def set_url(self, url) :
        self._url = url

    def get_label(self):
        """ Returns the label of the album """
        if self._label is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self.set_label(_get_label(self._xml.getElementsByTagName('label')))
        return self._label

    def set_label(self, label):
        self._label = label

    def get_release_date(self):
        """ Returns the release date of the album """
        if self._release_date is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self._release_date = _extract(self._xml, 'releaseDate')
        return self._release_date

    def set_release_date(self, release_date):
        self._release_date = release_date

    def get_added_date(self):
        """ Returns the added date of the album """
        if self._added_date is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self._added_date = _extract(self._xml, 'addedDate')
        return self._added_date

    def set_added_date(self, added_date):
        self._added_date = added_date

    def get_artist(self):
        """ Returns the Artist of the album """
        if not self.artist:
            self.set_artist(_get_artist(self._xml.getElementsByTagName('artist')))
        return self.artist

    def set_artist(self, artist):
        """ Sets the Artist object of the track """
        self.artist = artist

    def get_image(self):
        """ Returns album image url """
        if self._image is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'releaseid': self.id})
            self._image = _extract(self._xml, 'release_small_image')
        return self._image

    def set_image(self, image):
        if image is None: image = ''
        self._image = image

    def get_tags(self, pageSize=10):
        """ Returns the tags of the album """
        if self._tags is None:
            self._tags = []
            xml = self._request(self._method + 'tags', True, {'releaseid': self.id})
            for node in xml.getElementsByTagName('tag') :
                self._tags.append(_get_tag(node))
        if self._tags:
            self._tags.sort()
        return self._tags[:pageSize]

    def get_tracks(self, pageSize=10):
        """ Returns the tracks of the album """
        if self._tracks is not None: return self._tracks

        results = []
        xml = self._request(self._method + 'tracks', True, {'releaseid': self.id, 'pageSize': str(pageSize)})
        for node in xml.getElementsByTagName('track'):
            if self.artist is None:
                self.set_artist(_get_artist(node.getElementsByTagName('artist')))
            track = _get_track(node, self, self.get_artist())
            results.append(track)
        self._tracks = results
        return self._tracks

    def get_similar(self, pageSize=10):
        """ Returns a list similar albums """
        if self._similar is not None: return self._similar

        results = []
        xml = self._request(self._method + 'recommend', True, {'releaseid': self.id, 'pageSize': str(pageSize)})
        for node in xml.getElementsByTagName('release'):
            album = _get_album(node, _get_artist(node.getElementsByTagName('artist')[0]))
            if self == album:
                continue #Same album!
            results.append(album)
        self._similar = results
        return self._similar


class Track(_BaseObject):
    """ A Bmat track. """
    def __init__(self, id, artist=None):
        _BaseObject.__init__(self, '/track/')

        if isinstance(artist, Artist):
            self.artist = artist
        else:
            self.artist = None
        self.id = id
        self.title = None
        self.artist = None
        self.album = None

        self._isrc = None
        self._url = None
        self._preview = 'http://api.7digital.com/1.2/track/preview?trackid=' + self.id
        self._image = None
        self._tags = None
        self._duration = None
        self._version = None
        self._explicit = None
        self._position = None

    def __repr__(self):
        return self.get_artist().get_name().encode('utf8') + ' - ' + self.get_title().encode('utf8')

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __ne__(self, other):
        return self.get_id() != other.get_id()

    def get_id(self):
        """ Returns the track id """
        return self.id

    def get_title(self):
        """ Returns the track title """
        if self.title is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self.title = _extract(self._xml, 'track', 1)
        return self.title

    def set_title(self, title):
        self.title = title

    def get_isrc(self):
        """ Returns the ISRC of the track """
        if self._isrc is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self._isrc = _extract(self._xml, 'isrc')
        return self._isrc

    def set_isrc(self, isrc):
        self._isrc = isrc

    def get_url(self):
        """ Returns the url of the track """
        if self._url is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self._url = _extract(self._xml, 'url')
        return self._url

    def set_url(self, url):
        self._url = url

    def get_audio(self):
        return self.get_preview()

    def get_preview(self):
        """ Returns the url of the track """
        if self._preview is None :
            if not self._xml : self._xml = self._request(self._method + 'preview', True, {'trackid': self.id})
            self._preview = _extract(self._xml, 'url')
        return self._preview

    def get_duration(self):
        """ Returns the duration of the track """
        if self._duration is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self._duration = _extract(self._xml, 'duration')
        return self._duration

    def set_duration(self, duration):
        self._duration = duration

    def get_position(self):
        """ Returns the track number in the release """
        if self._position is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self._position = _extract(self._xml, 'trackNumber')
        return self._position

    def set_position(self, track_number):
        self._position = track_number

    def get_explicit(self):
        """ Returns whether the track contains explicit content """
        if self._explicit is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self._explicit = _extract(self._xml, 'explicitContent')
        return self._explicit

    def set_explicit(self, explicit):
        self._explicit = explicit

    def get_version(self):
        """ Returns the version of the track """
        if self._version is None :
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'trackid': self.id})
            self._version = _extract(self._xml, 'version')
        return self._version

    def set_version(self, version):
        self._version = version

    def get_artist(self):
        """ Returns the Artist of the track """
        if not self.artist:
            self.set_artist(_get_artist(self._xml.getElementsByTagName('artist')))
        return self.artist

    def set_artist(self, artist):
        """ Sets the Artist object of the track """
        self.artist = artist

    def get_album(self):
        """ Returns the associated Album object """
        if not self.album:
            self.set_album(_get_album(self._xml.getElementsByTagName('release')))
        return self.album

    def set_album(self, album):
        """ Sets the Album object of the track """
        self.album = album


class Tag(_BaseObject):
    """ A Tag """
    def __init__(self, id):
        _BaseObject.__init__(self, '/tag/')

        self.id = id
        self.name = None
        self._url = None

    def __repr__(self):
       return self.get_name().encode('utf8')

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __ne__(self, other):
        return self.get_id() != other.get_id()

    def get_id(self):
        """ Returns the tag id """
        return self.id

    def get_name(self):
        """ Returns the tag name """
        if self.name is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'tagid': self.id})
            self.name = _extract(self._xml, 'name')
        return self.name

    def set_name(self, name):
        self.name = name

    def get_url(self):
        """ Returns the url of the tag"""
        if self._url is None:
            if not self._xml : self._xml = self._request(self._method + 'details', True, {'tagid': self.id})
            self._url = _extract(self._xml, 'url')
        return self._url

    def set_url(self, url):
        self._url = url

class Label(_BaseObject):
    """ A Label """
    def __init__(self, id):
        _BaseObject.__init__(self, '')

        self.id = id
        self.name = None

    def __repr__(self):
       return self.get_name().encode('utf8')

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __ne__(self, other):
        return self.get_id() != other.get_id()

    def get_id(self):
        """ Returns the label id """
        return self.id

    def get_name(self):
        """ Returns the label name """
        return self.name

    def set_name(self, name):
        self.name = name


class _Search(_BaseObject):
    """ An abstract class for search """

    def __init__(self, method, search_terms, xml_tag):
        _BaseObject.__init__(self, method)

        self._search_terms = search_terms
        self._xml_tag = xml_tag

        self._results_per_page = 10
        if self._search_terms.has_key('pageSize'):
            self._results_per_page = str(self._search_terms['pageSize'])
        else:
            self._search_terms['pageSize'] = str(self._results_per_page)
        self._last_page_index = 0
        self._hits = None

    def get_total_result_count(self):
        if self._hits is not None:
            return self._hits
        params = self._get_params()
        params['pageSize'] = '1'
        xml = self._request(self._method, True, params)
        hits = int(_extract(xml, 'totalItems'))
        self._hits = hits
        return self._hits

    def _get_params(self):
        params = {}
        for key in self._search_terms.keys():
            params[key] = self._search_terms[key]
        return params

    def _retrieve_page(self, page_index):
        """ Returns the xml nodes to process """
        params = self._get_params()

        if page_index != 0:
            #offset = self._results_per_page * page_index
            params["page"] = str(page_index)

        doc = self._request(self._method, True, params)
        return doc.getElementsByTagName(self._xml_tag)[0]

    def _retrieve_next_page(self):
        self._last_page_index += 1
        return self._retrieve_page(self._last_page_index)

    def has_results(self):
        return self.get_total_result_count() > (self._results_per_page * self._last_page_index)

    def get_next_page(self):
        master_node = self._retrieve_next_page()
        return self._get_results(master_node)

    def get_page(self, page=0):
        if page < 0: page = 0
        if page > 0: page = page-1

        master_node = self._retrieve_page(page)
        return self._get_results(master_node)


class ArtistSearch(_Search):
    """ Search artists """

    def __init__(self, query):
        _Search.__init__(self, '/artist/search', {'q': query}, 'searchResults')

    def _get_results(self, master_node):
        results = []
        for node in master_node.getElementsByTagName('artist'):
            artist = _get_artist(node)
            results.append(artist)
        return results


class ArtistBrowse(_Search):
    """ Browse artists """

    def __init__(self, letter):
        _Search.__init__(self, '/artist/browse', {'letter': letter}, 'artists')

    def _get_results(self, master_node):
        results = []
        for node in master_node.getElementsByTagName('artist'):
            artist = _get_artist(node)
            results.append(artist)
        return results


class AlbumSearch(_Search):
    """ Search albums """

    def __init__(self, query):
        _Search.__init__(self, '/release/search', {'q': query}, 'searchResults')

    def _get_results(self, master_node):
        results = []
        for node in master_node.getElementsByTagName('release'):
            artist = _get_artist(node.getElementsByTagName('artist')[0])
            album = _get_album(node, artist)
            results.append(album)
        return results

class AlbumCharts(_Search):
    """ Chart albums """

    def __init__(self, period, todate):
        _Search.__init__(self, '/release/chart', {'period': period, 'todate': todate}, 'chart')

    def _get_results(self, master_node):
        results = []
        for node in master_node.getElementsByTagName('release'):
            artist = _get_artist(node.getElementsByTagName('artist')[0])
            album = _get_album(node, artist)
            results.append(album)
        return results

class AlbumReleases(_Search):
    """ Release albums by date """

    def __init__(self, fromdate, todate):
        _Search.__init__(self, '/release/bydate', {'fromDate': fromdate, 'toDate': todate}, 'releases')

    def _get_results(self, master_node):
        results = []
        for node in master_node.getElementsByTagName('release'):
            artist = _get_artist(node.getElementsByTagName('artist')[0])
            album = _get_album(node, artist)
            results.append(album)
        return results

class TrackSearch(_Search):
    """ Search for tracks """

    def __init__(self, query):
        _Search.__init__(self, '/track/search', {'q': query}, 'searchResults')

    def _get_results(self, master_node):
        results = []
        for node in master_node.getElementsByTagName('track'):
            artist = _get_artist(node.getElementsByTagName('artist')[0])
            album = _get_album(node.getElementsByTagName('release')[0], artist)
            track = _get_track(node, album, artist)
            results.append(track)
        return results


def search_artist(query):
    """Search artists by query. Returns an ArtistSearch object.
    Use get_next_page() to retrieve sequences of results."""
    return ArtistSearch(query)

def browse_artists(letter):
    """Browse artists by letter [a..z]. Returns an ArtistBrowse object.
    Use get_next_page() to retrieve sequences of results."""
    return ArtistBrowse(letter)

def search_album(query):
    """Search albums by query. Returns the albumSearch object.
    Use get_next_page() to retrieve sequences of results."""
    return AlbumSearch(query)

def album_charts(period, todate):
    """Get chart albums in a given period of time """
    return AlbumCharts(period, todate)

def album_releases(fromdate, todate):
    """Get releases in a given period of time"""
    return AlbumReleases(fromdate, todate)

def search_track(query):
    """Search tracks by query. Returns a TrackSearch object.
    Use get_next_page() to retrieve sequences of results."""
    return TrackSearch(query)


# XML
def _extract(node, name, index = 0):
    """Extracts a value from the xml string"""
    try:
        nodes = node.getElementsByTagName(name)

        if len(nodes):
            if nodes[index].firstChild:
                return nodes[index].firstChild.data.strip()
            else:
                return None
    except:
        return None

def _extract_all(node, name, pageSize_count = None):
    """Extracts all the values from the xml string. It returns a list."""
    results = []
    for i in range(0, len(node.getElementsByTagName(name))):
        if len(results) == pageSize_count:
            break
        results.append(_extract(node, name, i))
    return results

def _get_artist(xml):
    artist_id = xml.getAttribute('id')
    artist = Artist(artist_id)
    artist.set_name(_extract(xml, 'name'))
    return artist

def _get_album(xml, artist):
    album_id = xml.getAttribute('id')
    album = Album(album_id)
    album.set_artist(artist)
    album.set_title(_extract(xml, 'title'))
    album.set_type(_extract(xml, 'type'))
    album.set_image(_extract(xml, 'image'))
    album.set_year(_extract(xml, 'year'))
    album.set_barcode(_extract(xml, 'barcode'))
    album.set_url(_extract(xml, 'url'))
    album.set_release_date(_extract(xml, 'releaseDate'))
    album.set_added_date(_extract(xml, 'addedDate'))
    #TODO price, formats, artist appears_as
    try :
        album.set_label(_get_label(xml.getElementsByTagName('label')[0])) #In some cases that are albums with no label (record company) attached! :-(
    except:
        pass
    return album

def _get_track(xml, album, artist):
    track_id = xml.getAttribute('id')
    track = Track(track_id)
    track.set_title(_extract(xml, 'title'))
    track.set_url(_extract(xml, 'url'))
    track.set_isrc(_extract(xml, 'isrc'))
    track.set_duration(_extract(xml, 'duration'))
    track.set_position(_extract(xml, 'trackNumber'))
    track.set_explicit(_extract(xml, 'explicitContent'))
    track.set_version(_extract(xml, 'version'))
    track.set_album(album)
    track.set_artist(artist)
    #TODO price, formats
    return track

def _get_tag(xml):
    tag = Tag(_extract(xml, 'text'))
    tag.set_name(tag.get_id())
    return tag

def _get_label(xml):
    label = ''
    try:
        label = Label(xml.getAttribute('id'))
        label.set_name(_extract(xml, 'name'))
    except:
        pass
    return label

# CACHE
def enable_caching(cache_dir = None):
    global __cache_dir
    global __cache_enabled

    if cache_dir == None:
        import tempfile
        __cache_dir = tempfile.mkdtemp()
    else:
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        __cache_dir = cache_dir
    __cache_enabled = True

def disable_caching():
    global __cache_enabled
    __cache_enabled = False

def is_caching_enabled():
    """Returns True if caching is enabled."""
    global __cache_enabled
    return __cache_enabled

def _get_cache_dir():
    """Returns the directory in which cache files are saved."""
    global __cache_dir
    global __cache_enabled
    return __cache_dir

def get_md5(text):
    """Returns the md5 hash of a string."""
    hash = md5()
    hash.update(text.encode('utf8'))
    return hash.hexdigest()
