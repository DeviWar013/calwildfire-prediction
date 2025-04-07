from src.wildfires.data_loader import load_california_boundary, load_wildfires
from src.wildfires.wildfire_utils import wildfire_process
from src.wildfires.plot_map import plot_wildfires

def main():
    #loading shapefiles
    ca_boundary = load_california_boundary()
    wildfires = load_wildfires()

    #data processing
    wildfires_cal = wildfire_process(wildfires,ca_boundary)

    #plotting the clipped geometry
    plot_wildfires(ca_boundary, wildfires_cal)

if __name__ == "__main__":
    main()