import re
import sys
from vllm.entrypoints.cli.main import main
if __name__ == '__main__':
    
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    for i, arg in enumerate(sys.argv):
        print(f"参数 {i}: {arg}")
    sys.exit(main())