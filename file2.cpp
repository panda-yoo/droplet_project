#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

const double PI = 3.141592653589793;

// simulation parameters
const int N = 3;      // number of particles
const double L = 50.0;  // box size
const double v = 0.03;  // particle speed
const double R = 1.0;   // interaction radius
const double eta = 0.2; // noise strength
const int steps = 20000; // simulation steps

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

void storeDat(const std::vector<Particle> &, int,
              const std::filesystem::path &);

// periodic distance
double distance2(double x1, double y1, double x2, double y2) {
  double dx = x1 - x2;
  double dy = y1 - y2;

  // periodic correction
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

  // initialization
  for (int i = 0; i < N; i++) {
    particles[i].x = L * unif(gen);
    particles[i].y = L * unif(gen);
    particles[i].theta = 2 * PI * unif(gen);
  }

  for (int step = 0; step < steps + 1; step++) {

    std::vector<double> new_theta(N);

    // compute new directions
    for (int i = 0; i < N + 1; i++) {

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

      new_theta[i] = avg + eta * noise(gen);
    }

    // update particles
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
    }

    // simple output
    if (step % 100 == 0) {
      std::cout << "step " << step << std::endl;
    }
    if (step % 10 == 0) {
      storeDat(particles, step, output_dir);
    }
  }
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