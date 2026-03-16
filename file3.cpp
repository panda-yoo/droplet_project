#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

const double PI = 3.141592653589793;

// particle parameters
const int N = 5;
const double L = 50.0;
const double v = 0.1;
const double R = 1.5;
const double eta = 0.2;
const int steps = 500;

// chemical field grid
const int Nx = 100;
const int Ny = 100;
const double D = 0.1;     // diffusion coefficient
const double alpha = 0.5; // chemotaxis strength

// random generators
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> unif(0.0, 1.0);
std::uniform_real_distribution<double> noise(-0.5, 0.5);

struct Particle {
  double x;
  double y;
  double theta;
};

// periodic distance
double distance2(double x1, double y1, double x2, double y2) {
  double dx = x1 - x2;
  double dy = y1 - y2;

  if (dx > L / 2)
    dx -= L;
  if (dx < -L / 2)
    dx += L;
  if (dy > L / 2)
    dy -= L;
  if (dy < -L / 2)
    dy += L;

  return dx * dx + dy * dy;
}
void storeDat(const std::vector<Particle> &particles, int step,
              const std::filesystem::path &path) {
  const std::filesystem::path filename =
      path / (std::string("at") + std::to_string(step) + ".dat");
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Failed to open output file: " << filename << std::endl;
    return;
  }

  file << 'x' << ' ' << 'y' << std::endl;
  // compute new directions
  for (int i = 0; i < N; i++) {
    file << particles[i].x << ' ' << particles[i].y << std::endl;
  }
}
int main() {

  std::vector<Particle> particles(N);

  const std::filesystem::path output_dir("/work/vicsek_model/output");
  if (!std::filesystem::create_directories(output_dir) &&
      !std::filesystem::exists(output_dir)) {
    std::cerr << "Failed to create output directory: " << output_dir
              << std::endl;
    return 1;
  }
  std::cout << "Writing output to: " << output_dir << std::endl;

  // chemical field
  std::vector<std::vector<double>> C(Nx, std::vector<double>(Ny, 0.0));
  std::vector<std::vector<double>> Cnew(Nx, std::vector<double>(Ny, 0.0));

  // initialize particles
  for (int i = 0; i < N; i++) {
    particles[i].x = L * unif(gen);
    particles[i].y = L * unif(gen);
    particles[i].theta = 2 * PI * unif(gen);
  }

  for (int step = 0; step < steps; step++) {

    std::vector<double> new_theta(N);

    // --- VICSEK ALIGNMENT ---
    for (int i = 0; i < N; i++) {
      double sumx = 0;
      double sumy = 0;

      for (int j = 0; j < N; j++) {
        double d2 = distance2(particles[i].x, particles[i].y, particles[j].x,
                              particles[j].y);

        if (d2 < R * R) {
          sumx += cos(particles[j].theta);
          sumy += sin(particles[j].theta);
        }
      }

      double avg = atan2(sumy, sumx);

      // map particle to chemical grid
      int ix = particles[i].x / L * Nx;
      int iy = particles[i].y / L * Ny;

      if (ix <= 0)
        ix = 1;
      if (ix >= Nx - 1)
        ix = Nx - 2;
      if (iy <= 0)
        iy = 1;
      if (iy >= Ny - 1)
        iy = Ny - 2;

      // chemical gradient
      double gradx = (C[ix + 1][iy] - C[ix - 1][iy]) * 0.5;
      double grady = (C[ix][iy + 1] - C[ix][iy - 1]) * 0.5;

      double chem_angle = atan2(grady, gradx);

      new_theta[i] =
          avg + eta * noise(gen) + alpha * (chem_angle - particles[i].theta);
    }

    // --- MOVE PARTICLES ---
    for (int i = 0; i < N; i++) {

      particles[i].theta = new_theta[i];

      particles[i].x += v * cos(particles[i].theta);
      particles[i].y += v * sin(particles[i].theta);

      // periodic boundaries
      if (particles[i].x < 0)
        particles[i].x += L;
      if (particles[i].x > L)
        particles[i].x -= L;

      if (particles[i].y < 0)
        particles[i].y += L;
      if (particles[i].y > L)
        particles[i].y -= L;

      // deposit chemical
      int ix = particles[i].x / L * Nx;
      int iy = particles[i].y / L * Ny;

      C[ix][iy] += 0.2;
    }

    // --- DIFFUSE CHEMICAL FIELD ---
    for (int i = 1; i < Nx - 1; i++) {
      for (int j = 1; j < Ny - 1; j++) {
        Cnew[i][j] = C[i][j] + D * (C[i + 1][j] + C[i - 1][j] + C[i][j + 1] +
                                    C[i][j - 1] - 4 * C[i][j]);
      }
    }

    // swap fields
    C.swap(Cnew);

    if (step % 500 == 0)
      std::cout << "step " << step << std::endl;
    if (step % 25 == 0) {
      storeDat(particles, step, output_dir);
    }
  }
}