"""Read private Kaggle notebook metadata and version-specific output."""
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
import zipfile


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


def download(ref, version, target, remote_file=None):
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    target = Path(target)
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = ref.split("/")
    request.version_number = int(version)
    request.file_path = remote_file or target.name
    limit = 120_000_000
    incoming = target.with_suffix(".download")
    with api.build_kaggle_client() as client:
        with client.kernels.kernels_api_client.download_kernel_output(request) as response:
            wrapped = unquote(urlparse(response.url).path).endswith("/" + target.name + ".zip")
            with incoming.open("xb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if output.tell() + len(block) > limit:
                        raise ValueError("Output exceeds size limit")
                    output.write(block)
    if wrapped:
        with zipfile.ZipFile(incoming) as archive:
            if archive.namelist() != [target.name] or archive.getinfo(target.name).file_size > limit:
                raise ValueError("Unexpected wrapped output")
            with archive.open(target.name) as source, target.open("xb") as output:
                from shutil import copyfileobj
                copyfileobj(source, output)
    else:
        incoming.rename(target)
    print("VERSIONED_OUTPUT_READY")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(download(*sys.argv[2:]) if sys.argv[1] == "download" else main(sys.argv[1]))
    except Exception:
        sys.exit(1)
