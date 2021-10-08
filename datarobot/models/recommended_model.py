import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.model import Model

from ..utils import encode_utf8_if_py2


class ModelRecommendation(APIObject):
    """ A collection of information about a recommended model for a project.

    Attributes
    ----------
    project_id : str
        the id of the project the model belongs to
    model_id : str
        the id of the recommended model
    recommendation_type : str
        the type of model recommendation
    """

    _base_recommended_path_template = "projects/{}/recommendedModels/"
    _converter = t.Dict(
        {
            t.Key("project_id"): t.String,
            t.Key("model_id"): t.String,
            t.Key("recommendation_type"): t.String,
        }
    ).ignore_extra("*")

    def __init__(self, project_id, model_id, recommendation_type):
        self.project_id = project_id
        self.model_id = model_id
        self.recommendation_type = recommendation_type

    def __repr__(self):
        return encode_utf8_if_py2(
            u"ModelRecommendation({}, {}, {})".format(
                self.project_id, self.model_id, self.recommendation_type
            )
        )

    @classmethod
    def get(cls, project_id, recommendation_type=None):
        """
        Retrieves the default or specified by recommendation_type recommendation.

        Parameters
        ----------
        project_id : str
            The project's id.
        recommendation_type : enums.RECOMMENDED_MODEL_TYPE
            The type of recommendation to get. If None, returns the default recommendation.

        Returns
        -------
        recommended_model : ModelRecommendation

        """
        if recommendation_type is None:
            url = cls._base_recommended_path_template.format(project_id) + "recommendedModel/"
            return cls.from_location(url)
        else:
            recommendations = cls.get_all(project_id)
            return cls.get_recommendation(recommendations, recommendation_type)

    @classmethod
    def get_all(cls, project_id):
        """
        Retrieves all of the current recommended models for the project.


        Parameters
        ----------
        project_id : str
            The project's id.

        Returns
        -------
        recommended_models : list of ModelRecommendation
        """
        url = cls._base_recommended_path_template.format(project_id)
        response = ModelRecommendation._server_data(url)
        return [ModelRecommendation.from_server_data(data) for data in response]

    @classmethod
    def get_recommendation(cls, recommended_models, recommendation_type):
        """
        Returns the model in the given list with the requested type.


        Parameters
        ----------
        recommended_models : list of ModelRecommendation
        recommendation_type : enums.RECOMMENDED_MODEL_TYPE
            the type of model to extract from the recommended_models list

        Returns
        -------
        recommended_model : ModelRecommendation or None if no model with the requested type exists
        """

        return next(
            (
                model
                for model in recommended_models
                if model.recommendation_type == recommendation_type
            ),
            None,
        )

    def get_model(self):
        """
        Returns the Model associated with this ModelRecommendation.

        Returns
        -------
        recommended_model : Model
        """
        return Model.get(self.project_id, self.model_id)
