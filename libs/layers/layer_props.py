from pydantic import BaseModel
from pydantic.fields import ModelField
from pydantic.main import Model


def auto_property(klass):
    """Generate property from class constructor arguments and its annotation.
    """
    def _():
        klass = int()
        fields = {}
        annotations = {}
        for key, annot in int.__init__.__annotations__.items():
            fields[key] = ModelField(name=key, type_=)
            annotations[key] = annot
        klass.Property = type(
            'Property', BaseModel,
            {
                '__fields__': fields,
                '__annotations__': annotations,
            }
        )
        return klass
    return _
