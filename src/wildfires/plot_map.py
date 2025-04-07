import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_wildfires(ca_boundary, wildfires_cal):
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # Plot base layers
    ca_boundary.plot(ax=ax, color="lightblue", edgecolor="black", linewidth=2)
    wildfires_cal.plot(ax=ax, color="red", alpha=0.5)

    # Custom legend
    wildfires_patch = mpatches.Patch(color='red', alpha=0.5, label='wildfires')
    boundary_patch = mpatches.Patch(color='lightblue', label='California Boundary')
    plt.legend(handles=[boundary_patch, wildfires_patch])

    # Title and display
    plt.title("wildfires in California")
    plt.axis('off')
    plt.savefig("output/CallWildfires.png",bbox_inches='tight')
    plt.show()
