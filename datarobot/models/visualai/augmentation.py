import trafaret as t

from ..api_object import APIObject
from .images import Image

__all__ = ["ImageAugmentationOptions", "ImageAugmentationList", "ImageAugmentationSample"]


class ImageAugmentationOptions(APIObject):
    """A List of all supported Image Augmentation Transformations for a project.
    Includes additional information about minimum, maximum, and default values
    for a transformation.

    Attributes
    ----------
    name: string
        The name of the augmentation list
    project_id: string
        The project containing the image data to be augmented
    min_transformation_probability: float
        The minimum allowed value for transformation probability.
    current_transformation_probability: float
        Default setting for probability that each transformation will be applied to an image.
    max_transformation_probability: float
        The maximum allowed value for transformation probability.
    min_number_of_new_images: int
         The minimum allowed number of new rows to add for each existing row
    current_number_of_new_images: int
         The default number of new rows to add for each existing row
    max_number_of_new_images: int
         The maximum allowed number of new rows to add for each existing row
    transformations: array
        List of transformations to possibly apply to each image
    """

    _get_path = "imageAugmentationOptions/{pid}"
    _converter = t.Dict(
        {
            t.Key("project_id"): t.String(),
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("min_transformation_probability"): t.Float(),
            t.Key("max_transformation_probability"): t.Float(),
            t.Key("current_transformation_probability"): t.Float(),
            t.Key("min_number_of_new_images"): t.Int(),
            t.Key("max_number_of_new_images"): t.Int(),
            t.Key("current_number_of_new_images"): t.Int(),
            t.Key("transformations", optional=True): t.List(t.Dict().allow_extra("*")),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id,
        name,
        project_id,
        min_transformation_probability,
        current_transformation_probability,
        max_transformation_probability,
        min_number_of_new_images,
        current_number_of_new_images,
        max_number_of_new_images,
        transformations=None,
    ):
        self.id = id
        self.name = name
        self.project_id = project_id
        self.min_transformation_probability = min_transformation_probability
        self.current_transformation_probability = current_transformation_probability
        self.max_transformation_probability = max_transformation_probability
        self.min_number_of_new_images = min_number_of_new_images
        self.current_number_of_new_images = current_number_of_new_images
        self.max_number_of_new_images = max_number_of_new_images
        self.transformations = transformations

    def __repr__(self):
        return (
            "datarobot.models.visualai.ImageAugmentationOptions("
            "id={0.id}, "
            "name={0.name}, "
            "project_id={0.project_id}, "
            "min_transformation_probability={0.min_transformation_probability}, "
            "max_transformation_probability={0.max_transformation_probability}, "
            "current_transformation_probability={0.current_transformation_probability}, "
            "min_number_of_new_images={0.min_number_of_new_images})"
            "max_number_of_new_images={0.max_number_of_new_images})"
            "current_number_of_new_images={0.current_number_of_new_images})"
        ).format(self)

    @classmethod
    def get(cls, project_id):
        """
        Returns a list of all supported transformations for the given
        project

        :param project_id: sting
            The id of the project for which to return the list of supported transformations.

        :return:
          ImageAugmentationOptions
           A list containing all the supported transformations for the project.
        """
        path = cls._get_path.format(pid=project_id)
        server_data = cls._client.get(path)
        return cls.from_server_data(server_data.json())


class ImageAugmentationList(APIObject):

    """A List of Image Augmentation Transformations

    Attributes
    ----------
    name: string
        The name of the augmentation list
    project_id: string
        The project containing the image data to be augmented
    feature_name: string (optional)
        name of the feature that the augmentation list is associated with
    in_use: boolean
        Whether this is the list that will passed in to every blueprint during blueprint generation
        before autopilot
    initial_list: boolean
        True if this is the list to be used during training to produce augmentations
    transformation_probability: float
        Probability that each transformation will be applied to an image.  Value should be
        between 0.01 - 1.0.
    number_of_new_images: int
         Number of new rows to add for each existing row
    transformations: array
        List of transformations to possibly apply to each image
    """

    _create_path = "imageAugmentationLists/"
    _converter = t.Dict(
        {
            t.Key("project_id"): t.String(),
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("feature_name", optional=True): t.String(),
            t.Key("in_use", optional=True): t.Bool(),
            t.Key("initial_list", optional=True): t.Bool(),
            t.Key("transformation_probability", optional=True): t.Float(),
            t.Key("number_of_new_images", optional=True, default=1): t.Int(),
            t.Key("transformations", optional=True): t.List(t.Dict().allow_extra("*")),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id,
        name,
        project_id,
        feature_name=None,
        in_use=False,
        initial_list=False,
        transformation_probability=0.0,
        number_of_new_images=1,
        transformations=None,
    ):
        self.id = id
        self.name = name
        self.project_id = project_id
        self.feature_name = feature_name
        self.in_use = in_use
        self.initial_list = initial_list
        self.transformation_probability = transformation_probability
        self.number_of_new_images = number_of_new_images
        self.transformations = transformations

    def __repr__(self):
        return (
            "datarobot.models.visualai.ImageAugmentationList("
            "aug_id={0.id}, "
            "name={0.name}, "
            "project_id={0.project_id}, "
            "feature_name={0.feature_name}, "
            "in_use={0.in_use}, "
            "initial_list={0.initial_list}, "
            "transformation_probability={0.transformation_probability}, "
            "number_of_new_images={0.number_of_new_images})"
        ).format(self)

    @classmethod
    def create(
        cls,
        name,
        project_id,
        feature_name=None,
        in_use=False,
        initial_list=False,
        transformation_probability=0.0,
        number_of_new_images=1,
        transformations=None,
    ):
        """
        create a new image augmentation list
        """
        data = {
            "name": name,
            "project_id": project_id,
            "feature_name": feature_name,
            "in_use": in_use,
            "initial_list": initial_list,
            "transformation_probability": transformation_probability,
            "number_of_new_images": number_of_new_images,
            "transformations": transformations,
        }
        server_data = cls._client.post(cls._create_path, data=data)
        list_id = server_data.json()["augmentationListId"]
        return cls.get(list_id)

    @classmethod
    def get(cls, list_id):
        path = cls._create_path + str(list_id) + "/"
        server_data = cls._client.get(path)
        return cls.from_server_data(server_data.json())

    @classmethod
    def delete(cls, list_id):
        path = cls._create_path + str(list_id) + "/"
        cls._client.delete(path)


class ImageAugmentationSample(APIObject):
    """
    A preview of the type of images that augmentations will create during training.

     Attributes
    ----------
    sample_id : ObjectId
        The id of the augmentation sample, used to group related images together
    image_id : ObjectId
        A reference to the Image which can be used to retrieve the image binary
    project_id : ObjectId
        A reference to the project containing the image
    original_image_id : ObjectId
        A reference to the original image that generated this image in the case of an augmented
        image.  If this is None it signifies this is an original image
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    """

    _compute_path = "imageAugmentationSamples/"
    _list_path = "imageAugmentationSamples/{sample_id}/"
    _converter = t.Dict(
        {
            t.Key("image_id"): t.String(),
            t.Key("project_id"): t.String(),
            t.Key("height"): t.Int(),
            t.Key("width"): t.Int(),
            t.Key("original_image_id", optional=True): t.String(),
            t.Key("sample_id", optional=True): t.String(),
        }
    ).ignore_extra("*")

    def __init__(self, image_id, project_id, height, width, original_image_id=None, sample_id=None):
        self.sample_id = sample_id
        self.image_id = image_id
        self.project_id = project_id
        self.original_image_id = original_image_id
        self.height = height
        self.width = width
        self.image = Image(image_id=image_id, project_id=project_id, height=height, width=width)

    def __repr__(self):
        return (
            "datarobot.models.visualai.ImageAugmentationSample("
            "image_id={0.image_id}, "
            "project_id={0.project_id}, "
            "height={0.height}, "
            "width={0.width}, "
            "original_image_id={0.original_image_id})"
            "sample_id={0.sample_id}, "
        ).format(self)

    @classmethod
    def compute(cls, augmentation_list, number_of_rows=5):
        """Start creation of image augmentation samples.

        Parameters
        ----------
        number_of_rows: int
            The number of rows from the original dataset to use as input data for the
            augmentation samples. Defaults to 5.
        augmentation_list: ImageAugmentationList
            An Image Augmentation list that specifies the transformations to apply to each image
            during augmentation.

        Returns
        -------
        str
            URL to check for image augmentation samples generation progress.

        Raises
        ------
        datarobot.errors.ClientError
            Server rejected creation due to client error. Most likely
            cause is bad invalid ``augmentation_list``.
        """
        post_data = augmentation_list.__dict__
        post_data["number_of_rows"] = number_of_rows
        r_data = cls._client.post(cls._compute_path, data=post_data)
        return r_data.headers["Location"]

    @classmethod
    def list(cls, sample_id):
        """Return a list of ImageAugmentationSample objects

         Parameters
         ----------
        sample_id: str
             Unique Id for the set of sample images
        """
        path = cls._list_path.format(sample_id=sample_id)

        result = cls._client.get(path)
        r_data = result.json()
        ret = []
        for sample_data in r_data.get("data", []):
            sample = cls.from_server_data(sample_data)
            sample.sample_id = sample_id
            ret.append(sample)
        return ret
