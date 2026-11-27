"""Output only the verified ACL result from the authenticated Kaggle SDK."""
import sys


def main(ref):
    from kaggle import api
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    owner, slug = ref.split("/")
    request = ApiGetDatasetRequest()
    request.owner_slug = owner
    request.dataset_slug = slug
    with api.build_kaggle_client() as client:
        dataset = client.datasets.dataset_api_client.get_dataset(request)
    if dataset.is_private is True and dataset.current_version_number == 1:
        print("PRIVATE_VERSION_1")
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception:
        sys.exit(1)
