# submit_all.sh
for model_name in "MT-MLP" "PretrainFT" "Decoupled"; do
  case $model_name in

    "MT-MLP")
      MODEL_CLASS="MultiTaskNN"
      for lr in 0.001 0.01; do
      for hidden_dim in 64 128 256; do
      for num_layers in 1 2 3 4; do
      for lambda_aux in 0.1 0.3; do

        PARAM_JSON=$(jq -nc \
          --arg name "$model_name" \
          --arg model "$MODEL_CLASS" \
          --argjson params \
            "{\"lr\": $lr, \"hidden_dim\": $hidden_dim, \"num_layers\": $num_layers, \"lambda_aux\": $lambda_aux}" \
          '{name: $name, model: $model, params: $params}'
        )
        sbatch run_model.sh "$PARAM_JSON"

      done; done; done; done
      ;; 

    "PretrainFT")
      MODEL_CLASS="MultiTaskNN_PretrainFinetuneExtended"
      for lr_pre in 0.01 0.1; do
      for lr_fine in 0.01 0.1; do
      for hidden_dim in 64 128; do
      for num_layers in 1 2; do
      for lambda_aux in 0.1; do
      for pre_epochs in 100; do
      for fine_epochs in 100; do

        PARAM_JSON=$(jq -nc \
          --arg name "$model_name" \
          --arg model "$MODEL_CLASS" \
          --argjson params \
            "{\"lr_pre\": $lr_pre, \"lr_fine\": $lr_fine, \"hidden_dim\": $hidden_dim, \"num_layers\": $num_layers, \"lambda_aux\": $lambda_aux, \"pre_epochs\": $pre_epochs, \"fine_epochs\": $fine_epochs}" \
          '{name: $name, model: $model, params: $params}'
        )
        sbatch run_model.sh "$PARAM_JSON"

      done; done; done; done; done; done; done
      ;;

    "Decoupled")
      MODEL_CLASS="MultiTaskNN_Decoupled"
      for lr in 0.01 0.1; do
      for hidden_dim in 64 128 256; do
      for num_layers in 1 2 3 4; do
      for lambda_aux in 0.01 0.1 0.3; do

        PARAM_JSON=$(jq -nc \
          --arg name "$model_name" \
          --arg model "$MODEL_CLASS" \
          --argjson params \
            "{\"lr\": $lr, \"hidden_dim\": $hidden_dim, \"num_layers\": $num_layers, \"lambda_aux\": $lambda_aux}" \
          '{name: $name, model: $model, params: $params}'
        )
        sbatch run_model.sh "$PARAM_JSON"

      done; done; done; done
      ;;

  esac
done
