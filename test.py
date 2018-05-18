from pprint import pprint
import time
import utiliviz as uv

def main():
    with uv.record([uv.CpuMon]) as r:
        print("HAHAHA")
        time.sleep(0.2)
    pprint(r.get_data())

if __name__ == '__main__':
    main()