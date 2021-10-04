import trafaret as t

from datarobot.models.api_object import APIObject

from ..utils import encode_utf8_if_py2
from ..utils.pagination import unpaginate


class PredictionServer(APIObject):
    """A prediction server can be used to make predictions

    Attributes
    ----------
    id : str
        the id of the prediction server
    url : str
        the url of the prediction server
    datarobot_key : str
        the `datarobot-key` header used in requests to this prediction server
    """

    _path = "predictionServers/"
    _converter = t.Dict(
        {
            t.Key("id", optional=True) >> "id": t.String(),
            t.Key("url") >> "url": t.String(allow_blank=True),
            t.Key("datarobot-key", optional=True) >> "datarobot_key": t.String(allow_blank=True),
        }
    ).allow_extra("*")

    def __init__(self, id=None, url=None, datarobot_key=None):
        self.id = id
        self.url = url
        self.datarobot_key = datarobot_key

    def __repr__(self):
        return encode_utf8_if_py2(u"{}({})".format(self.__class__.__name__, self.url or self.id))

    @classmethod
    def list(cls):
        """Returns a list of prediction servers a user can use to make predictions.

        .. versionadded:: v2.17

        Returns
        -------
        prediction_servers : list of PredictionServer instances
            Contains a list of prediction servers that can be used to make predictions.

        Examples
        --------
        .. code-block:: python

            prediction_servers = PredictionServer.list()
            prediction_servers
            >>> [PredictionServer('https://example.com')]
        """

        data = unpaginate(cls._path, {}, cls._client)
        return [cls.from_server_data(item) for item in data]
