# Point Cloud Registration Assignment

## Objective

The goal of this assignment is to implement a point cloud registration algorithm. You will be provided with two point clouds: a source and a target. Your task is to find the rigid transformation (rotation and translation) that aligns the source point cloud to the target point cloud.

## Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/malek-wahidi/AcademyPointClouds.git
   cd AcademyPointClouds
   ```

2. **Create your own GitHub repository and change the origin:**

   Do **not fork** the repository and **do not submit a Pull Request**.

   Instead, create a **new empty repository on your own GitHub account**, then update the `origin` of your cloned repository to point to it.

   Example:

   ```bash
   git remote remove origin
   git remote add origin https://github.com/<your-username>/<your-repository-name>.git
   git push -u origin main
   ```

   From this point on, you will push your work to **your own repository**.

3. **Set up the environment:**

   This project uses `uv` for package management.

   ```bash
   uv sync
   ```

4. **Implement your registration function:**

   Open the `registration.py` file. You will find a function called `register`. You need to implement your registration algorithm inside this function. The function takes two `open3d.geometry.PointCloud` objects as input and should return a **4×4 NumPy array representing the transformation matrix**.

5. **Run the code:**

   You can run the main script to test your implementation:

   ```bash
   uv run main.py
   ```

   This will load a sample point cloud, create a transformed version of it, and then call your `register` function to find the alignment. The script will also time how long your registration takes.

6. **Submit your solution:**

   Push your completed work to **your own GitHub repository**.

   To submit the assignment, **send me the link to your repository**.

   Your submission will be evaluated based on:

   * **Correctness:** How well your algorithm aligns the point clouds.
   * **Performance:** The execution time of your registration function.
   * **Code Quality:** The clarity and efficiency of your implementation.

## Troubleshooting

If you encounter errors related to Open3D visualization (such as `GLFW Error: Wayland: The platform does not support setting the window position` or `Failed creating OpenGL window`) on Linux systems using Wayland, try running the script with X11 compatibility:

```bash
XDG_SESSION_TYPE=x11 uv run main.py
```

This can help resolve issues with OpenGL window creation in some desktop environments.
