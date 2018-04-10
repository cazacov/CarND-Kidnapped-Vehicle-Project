/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define EPS 1E-6

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 20;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (auto &particle : particles)  {

    if (fabs(yaw_rate) > EPS) {
      particle.x += velocity  / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y += velocity  / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }
    else {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }

    // add random noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (auto &observation : observations)
  {
    double min_distance = std::numeric_limits<double>::max();  // some big number

    for (auto &prediction : predicted) {
      double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
      if (distance < min_distance)
      {
        min_distance = distance;
        observation.id = prediction.id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++)
  {
    Particle particle = particles[i];

    // Predict landmarks in car's sensor range
    std::vector<LandmarkObs> predictions;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      double lx = map_landmarks.landmark_list[j].x_f;
      double ly = map_landmarks.landmark_list[j].y_f;
      double lid = map_landmarks.landmark_list[j].id_i;

      double distance = dist(lx, ly, particle.x, particle.y);
      if (distance <= sensor_range) {
        LandmarkObs prediction;
        prediction.id = lid;
        prediction.x = lx;
        prediction.y = ly;
        predictions.push_back(prediction);
      }
    }

    // Map observations in particles's coordinate system
    std::vector<LandmarkObs> obs;

    for (int j = 0; j < observations.size(); j++)
    {
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;

      double theta = particle.theta;

      double trans_x = particle.x + obs_x * cos(theta) - obs_y * sin(theta);
      double trans_y = particle.y + obs_x * sin(theta) + obs_y * cos(theta);

      LandmarkObs trans_obs = LandmarkObs();
      trans_obs.id = observations[j].id;
      trans_obs.x = trans_x;
      trans_obs.y = trans_y;
      obs.push_back(trans_obs);
    }

    // find nearest observations
    dataAssociation(predictions, obs);

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    // Caluclate particle's weight
    double weight = 1.0;

    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();

    for (int j = 0; j < obs.size(); j++) {

      LandmarkObs observation = obs[j];

      LandmarkObs prediction;
      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == observation.id) {
          prediction = predictions[k];
          break;
        }
      }

      particles[i].associations.push_back(prediction.id);
      particles[i].sense_x.push_back(observation.x);
      particles[i].sense_y.push_back(observation.y);

      double dx = observation.x - prediction.x;
      double dy = observation.y - prediction.y;

      // Calculate Multivariate-Gaussian probability density
      double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
      double exponent = dx * dx / (2 * sig_x * sig_x) + dy * dy / (2 * sig_y * sig_y);

      double multi_prob = gauss_norm * exp(-exponent);

      weight *= multi_prob;
    }
    particles[i].weight = weight;
  }

  // normalize weights

  double w_sum = 0;
  for (int i = 0; i < num_particles; i++) {
    w_sum +=  particles[i].weight;
  }

  // find best article
  double highest_weight = -1.0;
  Particle best_particle;
  int best_index;
  for (int i = 0; i < num_particles; ++i) {
    if (particles[i].weight > highest_weight) {
      highest_weight = particles[i].weight;
      best_particle = particles[i];
      best_index = i;
    }
  }

  weights.clear();
  if (w_sum > 0) {
    for (int i = 0; i < num_particles; i++) {
      particles[i].weight /= w_sum;
      weights.push_back(particles[i].weight);
    }
  }
  else {
    // That should not happen
    for (int i = 0; i < num_particles; i++) {
      particles[i].weight = 0;
      weights.push_back(particles[i].weight);
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


  // Use resampling wheel algorithm

  uniform_int_distribution<int> dist_i(0, num_particles - 1);
  int index = dist_i(gen);

  double max_weight = 0;
  for (int i = 0; i < num_particles; i++)
  {
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }
  uniform_real_distribution<double> dist_r(0, 2 * max_weight);

  std::vector<Particle> new_particles;

  double beta = 0;
  for (int i = 0; i < num_particles; i++)
  {
    beta += dist_r(gen);
    while(beta > particles[index].weight)
    {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
