/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <stdlib.h>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_real_distribution;
using namespace std;
std::default_random_engine gen;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;  
  
  long double std_x;
  long double std_y;
  long double std_theta;
  
  x = x;
  y = y;
  theta =theta;
  
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  Particle p;
  
  for (int i = 0; i < num_particles; ++i) {

    Particle p = {i, 
                  dist_x(gen),
                  dist_y(gen),
                  dist_theta(gen),
                  1, {}, {}, {}};   

    particles.push_back(p);
    weights.push_back(1.0);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  
    long double std_x;
    long double std_y;
    long double std_theta;
  
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2]; 

	normal_distribution<double> dist_x(0.0, std_x);
	normal_distribution<double> dist_y(0.0, std_y);
	normal_distribution<double> dist_theta(0.0, std_theta);
  
    for (int i = 0; i < num_particles; ++i){
      if (fabs(yaw_rate) < 0.00001)
      {
        particles[i].x += velocity * delta_t * cos(particles[i].theta);
        particles[i].y += velocity * delta_t * sin(particles[i].theta);  
      }
      else
      {
        particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
        particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
        particles[i].theta += yaw_rate * delta_t;   
      }
    
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  for (unsigned int i = 0; i < observations.size(); i++) {
    long double obsx = observations[i].x;
    long double obsy = observations[i].y;
    long double minimum_distance = numeric_limits<long double>::max();
    long double minimum_predicted_landmark = -1;

    for (unsigned int j = 0; j < predicted.size(); j++) {
      long double predict_x = predicted[j].x;
      long double predict_y = predicted[j].y;
      long double current_distance = dist(obsx, obsy, predict_x, predict_y);
      if (current_distance > minimum_distance) {
		continue;
      }
      else{
        minimum_distance = current_distance;
        minimum_predicted_landmark = j;        
      }
    }
    observations[i].id = minimum_predicted_landmark;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  long double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  long double sig_x = 2 * pow(std_landmark[0], 2);
  long double sig_y = 2* pow(std_landmark[1], 2);

  for (unsigned int i = 0; i < num_particles; i++) {
    long double x_part = particles[i].x; 
    long double y_part = particles[i].y; 
    long double theta_part = particles[i].theta; 

    vector<LandmarkObs> landmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double landmark_x =  map_landmarks.landmark_list[j].x_f;
      double landmark_y =  map_landmarks.landmark_list[j].y_f;
      long double distance = dist(landmark_x, landmark_y, particles[i].x, particles[i].y);
      
      if (distance >= sensor_range){
        continue;
      }
      else {
        landmarks.push_back(LandmarkObs {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }      

    vector<LandmarkObs> transformed_observations;
    for (unsigned int k = 0; k < observations.size(); k++) {

      LandmarkObs transformed_observation;
      transformed_observation.id = observations[k].id;
      transformed_observation.x = observations[k].x*cos(theta_part) - observations[k].y*sin(theta_part) + x_part;
      transformed_observation.y = observations[k].x*sin(theta_part) + observations[k].y*cos(theta_part) + y_part;
      transformed_observations.push_back(transformed_observation);

      dataAssociation(landmarks, transformed_observations);

      long double particle_probability = 1.0;      
      for (unsigned int k = 0; k < transformed_observations.size(); k++) {

        LandmarkObs transformed_observation = transformed_observations[k];
        particle_probability = particle_probability * gauss_norm * exp(-(pow(transformed_observation.x - landmarks[transformed_observation.id].x, 2.0)/sig_x + pow(transformed_observation.y - landmarks[transformed_observation.id].y, 2.0) / sig_y ));
      }
      
      particles[i].weight = particle_probability;
      weights[i] = particle_probability;
    }
  }
}

void ParticleFilter::resample() {
  
  vector<Particle> resampled;
  vector<long double> weights;
  
  long double max_weight = std::numeric_limits<long double>::min();
  
  for (unsigned int i = 0; i < num_particles; i++){
    weights.push_back(particles[i].weight);
    if (particles[i].weight < max_weight){
		continue;
    }
    else {
      max_weight = particles[i].weight;      
    }
  }

  std::uniform_real_distribution<long double> uniformRealDist(0.0, max_weight);
  std::uniform_int_distribution<int> uniformIntDist(0.0, num_particles - 1.0);
  int index = uniformIntDist(gen);

  long double beta = 0.0;
  long double mw = *std::max_element(std::begin(weights), std::end(weights));

  for (unsigned int i = 0; i < num_particles; i++){
    beta = beta + uniformRealDist(gen) * 2 * mw;

    while (beta > weights[index]){
      beta = beta - weights[index];
      index = (index + 1) % num_particles;
    }
      
    resampled.push_back(particles[index]);
  }

  particles = resampled;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1); 
  return s;  
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  
  return s;
}