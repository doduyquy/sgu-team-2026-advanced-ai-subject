"""Return the latest version number of an owned private Kaggle notebook."""
import sys


def main(ref):
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    owner, slug = ref.split("/")
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
    if metadata.is_private is True and metadata.current_version_number > 0:
        print(f"PRIVATE_KERNEL_VERSION {metadata.current_version_number}")
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception:
        sys.exit(1)
