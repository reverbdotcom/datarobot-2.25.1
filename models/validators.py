"""
Trafaret validators that doesn't part of the class.
Separated to own file to improve readability and reusability.
"""

import trafaret as t

# Representation of a single feature impact
# featureName - feature we are describing
# impactUnnormalized - drop in the metric value comparing to base
# impactNormalized - normalized value so that highest drop = 1.0
# redundantWith - one of 2 options
# - None if redundant detection wasn't run or
# - feature name that makes this one redundant
single_feature_impact_trafaret = t.Dict(
    {
        t.Key("impactUnnormalized"): t.Float,
        t.Key("impactNormalized"): t.Float,
        t.Key("redundantWith"): t.Or(t.Null, t.String),
        t.Key("featureName"): t.String,
    }
).ignore_extra("*")

# Feature impact data for regression, binary classification and aggregated score for multiclass
# ranRedundancyDetection - shows if redundancy detection was run for the model
# featureImpacts - list of impacts for each feature
feature_impact_trafaret = t.Dict(
    {
        t.Key("count"): t.Int,
        t.Key("ranRedundancyDetection"): t.Bool,
        t.Key("rowCount"): t.Or(t.Int, t.Null),
        t.Key("shapBased"): t.Bool,
        t.Key("featureImpacts"): t.List(single_feature_impact_trafaret),
    }
).ignore_extra("*")

# Feature impact data for each class in multiclass model
# classFeatureImpacts - list of records for each class, with 2 keys
# - class - target class name
# - featureImpacts - list of impacts of each feature for the class
multiclass_feature_impact_trafaret = t.Dict(
    {
        t.Key("ranRedundancyDetection"): t.Bool,
        t.Key("classFeatureImpacts"): t.List(
            t.Dict(
                {
                    t.Key("featureImpacts"): t.List(single_feature_impact_trafaret),
                    t.Key("class"): t.String,
                }
            ).ignore_extra("*")
        ),
    }
).ignore_extra("*")
