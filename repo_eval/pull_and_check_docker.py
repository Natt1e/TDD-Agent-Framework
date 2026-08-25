import os
import subprocess
import time

USER_NAME = "Natt1e"

def run_pytest(docker_image, pytest_args=None, memory=None, cpus=None, timeout=400) -> dict:
    """
    Run pytest in a Docker container.
    
    Args:
        docker_image: Docker image name/ID
        pytest_args: pytest arguments (default: ["-x", "-q", "--disable-warnings"])
        memory: Memory limit for container (e.g., "512m", "1g")
        cpus: CPU limit for container (e.g., "2", "0.5")
        timeout: Timeout in seconds for the docker command (default: 400)
    
    Return a dictionary containing the results.
    """
    docker_image = docker_image.strip()


    pytest_args = pytest_args or ["-x", "-q", "--disable-warnings"]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["MPLBACKEND"] = "Agg"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    proxy_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ]
    for var in proxy_vars:
        env[var] = ""

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-w",
        "/testbed",
        "-e",
        f"HF_ENDPOINT={env['HF_ENDPOINT']}",
        "-e",
        f"MPLBACKEND={env['MPLBACKEND']}",
        "-e",
        f"HF_HUB_OFFLINE={env['HF_HUB_OFFLINE']}",
        "-e",
        f"TRANSFORMERS_OFFLINE={env['TRANSFORMERS_OFFLINE']}",
        "-e",
        f"HF_DATASETS_OFFLINE={env['HF_DATASETS_OFFLINE']}",
    ]
    for var in proxy_vars:
        cmd.extend(["-e", f"{var}="])
    

    if memory:
        cmd.extend(["-m", memory])
    if cpus:
        cmd.extend(["--cpus", cpus])
    
    cmd.extend([
        docker_image,
        "pytest",
        *pytest_args,
    ])

    try:
        result = subprocess.run(
            cmd,
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "docker_image": docker_image,
        }

    except subprocess.TimeoutExpired as e:
        print(e)
        return {
            "status": "failed",
            "stdout": "Timeout",
            "stderr": "Timeout",
            "docker_image": docker_image,
        }
    except Exception as e:
        print(e)
        return {
            "status": "failed",
            "stdout": f"Error: {str(e)}",
            "stderr": f"Error: {str(e)}",
            "docker_image": docker_image,
        }
        
if __name__ == "__main__":
    # For double-blind review, USER_NAME is a place-holder Docker Hub namespace.
    # It will be replaced with the official maintainer namespace after acceptance.
    images = [
        f"{USER_NAME}/amazon-science_patchcore-inspection:v0",
        f"{USER_NAME}/carperai_trlx:v0",
        f"{USER_NAME}/deepmind_tracr:v0",
        f"{USER_NAME}/facebookresearch_omnivore:v0",
        f"{USER_NAME}/google_lightweight_mmm:v0",
        f"{USER_NAME}/leopard-ai_betty:v0",
        f"{USER_NAME}/lucidrains_imagen-pytorch:v0",
        f"{USER_NAME}/maxhumber_redframes:v0"
    ]
    failed_images = []

    for image in images:
        print(f"Running tests for Docker image: {image}")
        start_time = time.time()
        
        result = run_pytest(
            image,
            memory="12g",
            cpus="4.0",
            timeout=600
        )
        
        elapsed_time = time.time() - start_time
        status = result.get('status')
        
        print(f"Result for {image}: {status}")
        print(f"Execution time: {elapsed_time:.2f}s")
        
        if status not in ['success']: 
            failed_images.append(image)
            print(result)

    if not failed_images:
        print("✅ All images are ready!")
    else:
        print(f"❌ Warning: There are {len(failed_images)} images that failed the test, the list is as follows:")
        for img in failed_images:
            print(f"  - {img}")
    print("=" * 80)