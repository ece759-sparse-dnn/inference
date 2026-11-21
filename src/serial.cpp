#include <stdexcept>
#include <algorithm>

#include "sdnn/backend.hpp"

namespace sdnn {

void apply_activations(const std::vector<float>& left_layer,
						     std::vector<float>& right_layer,
					   const Dataset& d,
					   int left_layer_num){

	std::fill(right_layer.begin(), right_layer.end(), 0.0f);
	
	std::span<const std::int32_t> indptr  = d.layer_indptr(left_layer_num);
	std::span<const std::int32_t> indices = d.layer_indices(left_layer_num);
	std::span<const float>        values  = d.layer_values(left_layer_num);

	for(std::size_t i = 0; i < d.neurons; ++i){
		if(left_layer[i] == 0.0f){
			continue;
		}		
		int row_start = indptr[i];
		int row_end = indptr[i+1];
		for(std::size_t k = row_start; k < row_end; ++k){
			right_layer[indices[k]] += values[k] * left_layer[i];
		}
	}

	for(std::size_t i = 0; i < d.neurons; ++i){
		if(right_layer[i] > 0) right_layer[i] += d.bias;
		right_layer[i] = std::min(std::max(0.0f, right_layer[i]), d.ymax);
	}
}

std::vector<std::int32_t> infer_serial(const Dataset &d, int) {

}

} // namespace sdnn
