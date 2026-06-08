#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

// Function to generate and execute Python script for plotting
void plot_loss(const std::string& csv_file) {
    std::string python_script = 
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import sys\n"
        "\n"
        "try:\n"
        "    # Read the CSV file\n"
        "    df = pd.read_csv('" + csv_file + "')\n"
        "\n"
        "    # Filter out validation steps if they clutter the graph, or plot them separately\n"
        "    # Assuming standard format: step,loss,val_loss,...\n"
        "\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    plt.plot(df['step'], df['loss'], label='Training Loss', color='blue', alpha=0.7)\n"
        "    \n"
        "    # Plot validation loss if present and not -1\n"
        "    if 'val_loss' in df.columns:\n"
        "        val_data = df[df['val_loss'] > 0]\n"
        "        if not val_data.empty:\n"
        "             plt.plot(val_data['step'], val_data['val_loss'], label='Validation Loss', color='red', marker='o', linestyle='None')\n"
        "\n"
        "    plt.title('Training Loss vs Step')\n"
        "    plt.xlabel('Step')\n"
        "    plt.ylabel('Loss')\n"
        "    plt.legend()\n"
        "    plt.grid(True)\n"
        "    \n"
        "    # Save the plot\n"
        "    output_file = 'loss_graph_lr.png'\n"
        "    plt.savefig(output_file)\n"
        "    print(f'Graph saved to {output_file}')\n"
        "\n"
        "except Exception as e:\n"
        "    print(f'Error plotting: {e}')\n";

    // Save Python script to a temporary file
    std::ofstream script_file("plot_temp.py");
    script_file << python_script;
    script_file.close();

    // Execute the Python script
    std::cout << "Executing Python plotting script..." << std::endl;
    int result = std::system("python3 plot_temp.py");
    
    // Clean up
    std::remove("plot_temp.py");

    if (result == 0) {
        std::cout << "Plotting successful." << std::endl;
    } else {
        std::cerr << "Plotting failed." << std::endl;
    }
}

int main() {
    std::string csv_file = "/home/blubridge-029/tensor/Tensor-Implementation/bluwerp_prompt.csv";
    plot_loss(csv_file);
    return 0;
}
