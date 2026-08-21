#!/bin/bash

x_offset="${3:-0.5}"
y_offset="${4:-0.5}"
x_offset_sub="${5:-0.02}"
y_offset_sub="${6:-0.02}"
rot_angle="${7:-1.0}"
small_rot_angle="${8:-0.1}"
blurr_fwhm="${9:-2}"
N_frames="${10:-5}"
python_command="${11:-python}"
fabricate_script_path="${12:-prep_files.py}"
run_tests_script_path="${13:-run_tests.py}"

echo "Input_data: $1"
echo "Output_2: $2"
echo "x_offset: $x_offset"
echo "y_offset: $y_offset"
echo "x_offset_sub: $x_offset_sub"
echo "y_offset_sub: $y_offset_sub"
echo "rot_angle: $rot_angle"
echo "small_rot_angle: $small_rot_angle"
echo "blurr_fwhm: $blurr_fwhm"
echo "python_command: $python_command"
echo "N_frames: $N_frames"

directory_subfix_list=( 
    "_blurr_rot_shift"
    "_blurr_rot"
    "_blurr_small_rot_shift"
    "_blurr_small_rot"
    "_blurr_shift"
    "_blurr_sub_rot_shift"
    "_blurr_sub_small_rot_shift"
    "_blurr_sub_shift"
    "_rot_shift"
    "_small_rot_shift"
    "_shift"
    "_rot"
    "_small_rot"
    "_blurr"
    "_sub_rot_shift"
    "_sub_small_rot_shift"    
    "_sub_shift"
    "_control"
)

echo "Creating directories..."

# make the directories first
for subfix in "${directory_subfix_list[@]}"; do
    echo "Creating directory: ${2}_test${subfix}"
    mkdir -p "${2}_test${subfix}"
    echo "Creating directory: ${2}_results${subfix}"
    mkdir -p "${2}_results${subfix}"
done

echo "Done!"

# moving data to the test directories
for subfix in "${directory_subfix_list[@]}"; do
    echo "Moving data to ${2}_test${subfix}"
    files=("$1"/*.fits)
    if [ -e "${files[0]}" ]; then
        cp "${files[@]}" "${2}_test${subfix}/"
    else
        echo "Warning: no .fits files found in $1"
    fi
done

echo "Done!"

echo "Preparing fabricated frames..."

# --- blurred psf
# large shifts
${python_command} "$fabricate_script_path" "--x_offset" "$x_offset" "--y_offset" "$y_offset" "--rot_angle" "$rot_angle" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_rot_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_rot_shift" "./${2}_test_blurr_rot_shift"

${python_command} "$fabricate_script_path" "--x_offset" "$x_offset" "--y_offset" "$y_offset" "--rot_angle" "$small_rot_angle" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_small_rot_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_small_rot_shift" "./${2}_test_blurr_small_rot_shift"

${python_command} "$fabricate_script_path" "--rot_angle" "$rot_angle" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_rot" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_rot" "./${2}_test_blurr_rot"

${python_command} "$fabricate_script_path" "--rot_angle" "$small_rot_angle" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_small_rot" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_small_rot" "./${2}_test_blurr_small_rot"

${python_command} "$fabricate_script_path" "--x_offset" "$x_offset" "--y_offset" "$y_offset" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_shift" "./${2}_test_blurr_shift"

# sub-pixel shifts
${python_command} "$fabricate_script_path" "--x_offset" "$x_offset_sub" "--y_offset" "$y_offset_sub" "--rot_angle" "$rot_angle" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_sub_rot_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_sub_rot_shift" "./${2}_test_blurr_sub_rot_shift"

${python_command} "$fabricate_script_path" "--x_offset" "$x_offset_sub" "--y_offset" "$y_offset_sub" "--rot_angle" "$small_rot_angle" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_sub_small_rot_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_sub_small_rot_shift" "./${2}_test_blurr_sub_small_rot_shift"

${python_command} "$fabricate_script_path" "--x_offset" "$x_offset_sub" "--y_offset" "$y_offset_sub" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr_sub_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr_sub_shift" "./${2}_test_blurr_sub_shift"

${python_command} "$fabricate_script_path" "--blurr_fwhm" "$blurr_fwhm" "${2}_test_blurr" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_blurr" "./${2}_test_blurr"

# --- not blurred psf
# large shifts
${python_command} "$fabricate_script_path" "--x_offset" "$x_offset" "--y_offset" "$y_offset" "--rot_angle" "$rot_angle" "${2}_test_rot_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_rot_shift" "./${2}_test_rot_shift"

${python_command} "$fabricate_script_path" "--x_offset" "$x_offset" "--y_offset" "$y_offset" "--rot_angle" "$small_rot_angle" "${2}_test_small_rot_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_small_rot_shift" "./${2}_test_small_rot_shift"

${python_command} "$fabricate_script_path" "--rot_angle" "$rot_angle" "${2}_test_rot" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_rot" "./${2}_test_rot"

${python_command} "$fabricate_script_path" "--rot_angle" "$small_rot_angle" "${2}_test_small_rot" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_small_rot" "./${2}_test_small_rot"

${python_command} "$fabricate_script_path" "--x_offset" "$x_offset" "--y_offset" "$y_offset" "${2}_test_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_shift" "./${2}_test_shift"

# sub-pixel shifts
${python_command} "$fabricate_script_path" "--x_offset" "$x_offset_sub" "--y_offset" "$y_offset_sub" "--rot_angle" "$rot_angle" "${2}_test_sub_rot_shift" "$N_frames"
${python_command} "$run_tests_script_path" "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_sub_rot_shift" "./${2}_test_sub_rot_shift"
    
${python_command} "$fabricate_script_path" "--x_offset" "$x_offset_sub" "--y_offset" "$y_offset_sub" "--rot_angle" "$small_rot_angle" "${2}_test_sub_small_rot_shift" "$N_frames"
${python_command} "$run_tests_script_path" "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_sub_small_rot_shift" "./${2}_test_sub_small_rot_shift"


${python_command} "$fabricate_script_path" "--x_offset" "$x_offset_sub" "--y_offset" "$y_offset_sub" "${2}_test_sub_shift" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_sub_shift" "./${2}_test_sub_shift"

# control
${python_command} "$fabricate_script_path" "${2}_test_control" "$N_frames"
${python_command} $run_tests_script_path "--all_log_files" "--delete_files" "--N_cores" "40" "--output_dir" "./${2}_results_control" "./${2}_test_control"



echo "Done!"

echo "All tests completed!"



