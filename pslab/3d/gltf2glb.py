from pathlib import Path
from gltflib import GLTF
import sys

def gltf_to_glb(path, out):
  GLTF.load(path).export(out)
  

# kinda sus ngl, but im used to c
def main(argv, argc):
  
  path = Path(argv[1])

  assert path.suffix == ".gltf"

  npath = path.with_suffix(".glb")

  gltf_to_glb(path, npath)
  
  
if __name__  == "__main__":
  main(sys.argv, len(sys.argv))
