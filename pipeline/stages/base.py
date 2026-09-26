STAGE_REGISTRY: dict[str, type["PipelineStage"]] = {}


def register_stage(name):
    def decorator(cls):
        cls.name = name
        STAGE_REGISTRY[name] = cls
        return cls

    return decorator


class PipelineStage:
    name = "base"

    def run(self, context):
        raise NotImplementedError
