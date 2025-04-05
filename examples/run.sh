python3 register_arize_project.py && \
echo "Sleeping for 5 seconds..." && \
sleep 5 && \
python3 generate_samples_and_upload.py && \
echo "Sleeping for 60 seconds..." && \
sleep 60 && \
echo "Sleeping for 60 seconds..." && \
sleep 60 && \
echo "Sleeping for 60 seconds..." && \
sleep 60 && \
python3 run_and_upsert_evals.py
