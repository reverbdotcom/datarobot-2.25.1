import trafaret as t

UserBlueprintsHexColumnNameLookupEntry_ = t.Dict(
    {
        t.Key("colname"): t.String(allow_blank=False),
        t.Key("hex"): t.String(allow_blank=False),
        t.Key("project_id", optional=True): t.String(allow_blank=False),
    }
)

ParamValuePair_ = t.Dict(
    {
        t.Key("param"): t.String(allow_blank=False),
        t.Key("value", optional=True): t.Or(
            t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null()),
            t.List(t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null())),
        ),
    }
)

UserBlueprintsBlueprintTaskData_ = t.Dict(
    {
        t.Key("inputs"): t.List(t.String(allow_blank=False)),
        t.Key("output_method"): t.String(allow_blank=False),
        t.Key("output_method_parameters"): t.List(ParamValuePair_),
        t.Key("task_code"): t.String(allow_blank=False),
        t.Key("task_parameters"): t.List(ParamValuePair_),
        t.Key("x_transformations"): t.List(ParamValuePair_),
        t.Key("y_transformations"): t.List(ParamValuePair_),
    }
)

UserBlueprintsBlueprintTask_ = t.Dict(
    {
        t.Key("task_id"): t.String(allow_blank=False),
        t.Key("task_data"): UserBlueprintsBlueprintTaskData_,
    }
)

VertexContextItemInfo_ = t.Dict(
    {
        t.Key("inputs"): t.List(t.String(allow_blank=False)),
        t.Key("outputs"): t.List(t.String(allow_blank=False)),
    }
)

VertexContextItemMessages_ = t.Dict(
    {
        t.Key("errors", optional=True): t.List(t.String(allow_blank=False)),
        t.Key("warnings", optional=True): t.List(t.String(allow_blank=False)),
    }
)

VertexContextItem_ = t.Dict(
    {
        t.Key("task_id"): t.String(allow_blank=False),
        t.Key("information"): t.Or(VertexContextItemInfo_),
        t.Key("messages"): t.Or(VertexContextItemMessages_),
    }
)

UserBlueprint_ = t.Dict(
    {
        t.Key("blender"): t.Bool(),
        t.Key("blueprint_id"): t.String(allow_blank=False),
        t.Key("custom_task_version_metadata", optional=True): t.List(
            t.List(t.String(allow_blank=False))
        ),
        t.Key("diagram"): t.String(allow_blank=False),
        t.Key("features"): t.List(t.String(allow_blank=False)),
        t.Key("features_text"): t.String(allow_blank=True),
        t.Key("hex_column_name_lookup", optional=True): t.List(
            UserBlueprintsHexColumnNameLookupEntry_
        ),
        t.Key("icons"): t.List(t.Int()),
        t.Key("insights"): t.String(allow_blank=False),
        t.Key("is_time_series", default=False): t.Bool(),
        t.Key("model_type"): t.String(allow_blank=False),
        t.Key("project_id", optional=True): t.String(allow_blank=False),
        t.Key("reference_model", default=False): t.Bool(),
        t.Key("shap_support", default=False): t.Bool(),
        t.Key("supported_target_types"): t.List(t.String(allow_blank=False)),
        t.Key("supports_gpu", default=False): t.Bool(),
        t.Key("user_blueprint_id"): t.String(allow_blank=False),
        t.Key("user_id"): t.String(allow_blank=False),
        t.Key("blueprint", optional=True): t.List(UserBlueprintsBlueprintTask_),
        t.Key("vertex_context", optional=True): t.List(VertexContextItem_),
    }
).allow_extra("*")


UserBlueprintsInputType_ = t.Dict(
    {t.Key("type"): t.String(allow_blank=False), t.Key("name"): t.String(allow_blank=False)}
)

UserBlueprintsInputTypesResponse_ = t.Dict({t.Key("input_types"): t.List(UserBlueprintsInputType_)})


UserBlueprintAddedToMenuItem_ = t.Dict(
    {
        t.Key("blueprint_id"): t.String(allow_blank=False),
        t.Key("user_blueprint_id"): t.String(allow_blank=False),
    }
)

UserBlueprintAddToMenuResponse_ = t.Dict(
    {t.Key("added_to_menu"): t.List(UserBlueprintAddedToMenuItem_)}
)


UserBlueprintsValidateTaskParameter_ = t.Dict(
    {
        t.Key("message"): t.String(allow_blank=False),
        t.Key("param_name"): t.String(allow_blank=False),
        t.Key("value", optional=True): t.Or(
            t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null()),
            t.List(t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null())),
        ),
    }
)

UserBlueprintsValidateTaskParametersResponse_ = t.Dict(
    {t.Key("errors"): t.List(UserBlueprintsValidateTaskParameter_)}
)


UserBlueprintTaskCategoryItem_ = t.Dict(
    {
        t.Key("name"): t.String(allow_blank=False),
        t.Key("task_codes"): t.List(t.String(allow_blank=False)),
        t.Key("subcategories", optional=True): t.List(t.Dict().allow_extra("*")),
    }
).allow_extra("*")

UserBlueprintTaskArgumentDefinition_ = t.Dict(
    {
        t.Key("name"): t.String(allow_blank=False),
        t.Key("type"): t.String(allow_blank=False),
        t.Key("default", optional=True): t.Or(
            t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null()),
            t.List(t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null())),
        ),
        t.Key("values"): t.Or(
            t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null()),
            t.List(t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null())),
            t.Dict().allow_extra("*"),
        ),
        t.Key("tunable", optional=True): t.Bool(),
        t.Key("recommended", optional=True): t.Or(
            t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null()),
            t.List(t.Or(t.Int(), t.String(allow_blank=True), t.Bool(), t.Float(), t.Null())),
        ),
    }
).allow_extra("*")

UserBlueprintTaskArgument_ = t.Dict(
    {
        t.Key("key"): t.String(allow_blank=False),
        t.Key("argument"): t.Or(UserBlueprintTaskArgumentDefinition_),
    }
)

ColnameAndType_ = t.Dict(
    {
        t.Key("hex"): t.String(allow_blank=False),
        t.Key("colname"): t.String(allow_blank=False),
        t.Key("type"): t.String(allow_blank=False),
    }
)

TaskDocumentationUrl_ = t.Dict(
    {t.Key("documentation", optional=True): t.String(allow_blank=True)}
).allow_extra("*")

UserBlueprintTaskCustomTaskMetadata_ = t.Dict(
    {
        t.Key("id"): t.String(allow_blank=False),
        t.Key("version_major"): t.Int(),
        t.Key("version_minor"): t.Int(),
        t.Key("label"): t.String(allow_blank=False),
    }
)

UserBlueprintTask_ = t.Dict(
    {
        t.Key("task_code"): t.String(allow_blank=False),
        t.Key("label"): t.String(allow_blank=True),
        t.Key("description"): t.String(allow_blank=True),
        t.Key("arguments"): t.List(UserBlueprintTaskArgument_),
        t.Key("categories"): t.List(t.String(allow_blank=False)),
        t.Key("colnames_and_types", optional=True): t.Or(t.List(ColnameAndType_), t.Null),
        t.Key("icon"): t.Int(),
        t.Key("output_methods"): t.List(t.String(allow_blank=False)),
        t.Key("time_series_only"): t.Bool(),
        t.Key("url"): t.Or(
            t.Dict().allow_extra("*"), t.String(allow_blank=True), TaskDocumentationUrl_,
        ),
        t.Key("valid_inputs"): t.List(t.String(allow_blank=False)),
        t.Key("is_custom_task", optional=True): t.Bool(),
        t.Key("custom_task_id", optional=True): t.Or(t.String, t.Null),
        t.Key("custom_task_versions", optional=True): t.List(UserBlueprintTaskCustomTaskMetadata_),
        t.Key("supports_scoring_code", optional=True): t.Bool(),
    }
).allow_extra("*")

UserBlueprintTaskLookupEntry_ = t.Dict(
    {
        t.Key("task_code"): t.String(allow_blank=False),
        t.Key("task_definition"): t.Or(UserBlueprintTask_),
    }
)

UserBlueprintTasksResponse_ = t.Dict(
    {
        t.Key("categories"): t.List(UserBlueprintTaskCategoryItem_),
        t.Key("tasks"): t.List(UserBlueprintTaskLookupEntry_),
    }
)

UserBlueprintSharedRolesResponseValidator_ = t.Dict(
    {
        t.Key("share_recipient_type"): t.Enum("user", "group", "organization"),
        t.Key("role"): t.Enum("CONSUMER", "EDITOR", "OWNER"),
        t.Key("id"): t.String(allow_blank=False, min_length=24, max_length=24),
        t.Key("name"): t.String(allow_blank=False),
    }
)

UserBlueprintSharedRolesListResponseValidator_ = t.Dict(
    {
        t.Key("count", optional=True): t.Int(),
        t.Key("next", optional=True): t.URL,
        t.Key("previous", optional=True): t.URL,
        t.Key("data"): t.List(UserBlueprintSharedRolesResponseValidator_),
        t.Key("total_count", optional=True): t.Int(),
    }
)


UserBlueprintCatalogSearchItem_ = t.Dict(
    {
        t.Key("id"): t.String(),
        t.Key("catalog_name"): t.String(),
        t.Key("description", optional=True): t.String(),
        t.Key("info_creator_full_name"): t.String(),
        t.Key("last_modifier_name", optional=True): t.String(),
        t.Key("user_blueprint_id"): t.String(),
    }
).allow_extra("*")
