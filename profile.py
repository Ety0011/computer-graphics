import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

def get_profiles(timings_md):
    pattern = r'### (\w+)\s+File: (.+?)\s+Description: (.+?)\s+Average Time Total:(.*?)\s+Average Time FPS:(.*?)\s+Total Number of Triangles:(.*?)\s+Time per Triangle:(.*?)\s+(?=###|\Z)'
    matches = re.findall(pattern, timings_md, re.DOTALL)
    profiles = []
    for match in matches:
        profile_name = match[0].strip()
        file_attribute = match[1].strip()
        description = match[2].strip()
        total_time = match[3].strip()
        fps = match[4].strip()
        total_triangles = match[5].strip()
        time_per_triangle = match[6].strip()
        profiles.append({
            'name': profile_name,
            'file': file_attribute,
            'description': description,
            'total_time': total_time,
            'fps': fps,
            'total_triangles': total_triangles,
            'time_per_triangle': time_per_triangle,
            'section': match
        })
    return profiles

def build_profile(profile_name):
    print(f"\nBuilding profile: {profile_name}")
    result = subprocess.run(['make', profile_name.lower()], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building {profile_name}:\n{result.stderr}")
        return False
    return True

def run_profile(executable, args):
    cmd = [f'./{executable}'] + args
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {executable}:\n{result.stderr}")
        return None
    # Parse the output
    total_time_match = re.search(r'It took ([\d\.]+) seconds to render the image\.', result.stdout)
    fps_match = re.search(r'I could render at ([\d\.]+) frames per second\.', result.stdout)
    if total_time_match and fps_match:
        total_time = float(total_time_match.group(1))
        fps = float(fps_match.group(1))
        return total_time, fps
    else:
        print("Could not parse timing information from the output.")
        return None

def count_triangles_in_obj(file_path):
    count = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('f '):
                    count += 1
        return count
    except IOError:
        print(f"Error opening file {file_path}")
        return 0

def update_timings_md(timings_md, profile, avg_total_time, avg_fps, total_triangles, time_per_triangle):
    # Build the replacement section
    section = f"""### {profile['name']}

File: {profile['file']}
Description: {profile['description']}
Average Time Total: {avg_total_time}
Average Time FPS: {avg_fps}
Total Number of Triangles: {total_triangles}
Time per Triangle: {time_per_triangle}
"""
    # Replace the old section
    pattern = re.compile(rf'### {profile["name"]}\s+File: .+?\s+Description: .+?\s+Average Time Total:.*?\s+Average Time FPS:.*?\s+Total Number of Triangles:.*?\s+Time per Triangle:.*?(?=###|\Z)', re.DOTALL)
    timings_md = pattern.sub(section.strip(), timings_md)
    return timings_md

def plot_metrics(profiles):
    # Filter out profiles with missing data
    valid_profiles = [p for p in profiles if p['total_time'] and p['fps'] and p['time_per_triangle']]
    if not valid_profiles:
        print("No profiles with complete data to plot.")
        return

    names = [p['name'] for p in valid_profiles]
    avg_total_times = [float(p['total_time']) for p in valid_profiles]
    avg_fps_values = [float(p['fps']) for p in valid_profiles]
    time_per_triangle_values = [float(p['time_per_triangle']) for p in valid_profiles]

    # Plot Average Total Time
    plt.figure(figsize=(10, 6))
    plt.bar(names, avg_total_times, color='skyblue')
    plt.xlabel('Profiles')
    plt.ylabel('Average Total Time (s)')
    plt.title('Average Total Time per Profile')
    plt.savefig('average_total_time.png')
    plt.close()
    print("Saved plot: average_total_time.png")

    # Plot Average FPS
    plt.figure(figsize=(10, 6))
    plt.bar(names, avg_fps_values, color='lightgreen')
    plt.xlabel('Profiles')
    plt.ylabel('Average FPS')
    plt.title('Average FPS per Profile')
    plt.savefig('average_fps.png')
    plt.close()
    print("Saved plot: average_fps.png")

    # Plot Time per Triangle
    plt.figure(figsize=(10, 6))
    plt.bar(names, time_per_triangle_values, color='salmon')
    plt.xlabel('Profiles')
    plt.ylabel('Time per Triangle (s)')
    plt.title('Time per Triangle per Profile')
    plt.savefig('time_per_triangle.png')
    plt.close()
    print("Saved plot: time_per_triangle.png")

def main():
    with open('Bonus1/timings.md', 'r') as f:
        timings_md = f.read()

    profiles = get_profiles(timings_md)

    for profile in profiles:
        # Check if any of the fields are empty
        if (profile['total_time'] == '' or profile['fps'] == '' or
            profile['total_triangles'] == '' or profile['time_per_triangle'] == ''):
            print(f"\nProcessing profile: {profile['name']}")
            # Build the profile
            if not build_profile(profile['name']):
                continue
            # Extract filenames from the 'File:' attribute
            filenames = profile['file'].strip().split()
            # Count total number of triangles
            total_triangles = 0
            for filename in filenames:
                mesh_path = f"/home/leonardo/meshes/{filename}"
                triangles_in_file = count_triangles_in_obj(mesh_path)
                total_triangles += triangles_in_file
                print(f"Mesh {filename} has {triangles_in_file} triangles.")
            print(f"Total number of triangles: {total_triangles}")
            # Run the profile 10 times
            total_times = []
            fps_values = []
            executable = profile['name'].lower()
            for i in range(10):
                print(f"\nRun {i+1}/10 for profile {profile['name']}")
                result = run_profile(executable, filenames)
                if result:
                    total_time, fps = result
                    total_times.append(total_time)
                    fps_values.append(fps)
                else:
                    print(f"Skipping run {i+1} due to errors.")
            if total_times and fps_values and total_triangles > 0:
                avg_total_time = sum(total_times) / len(total_times)
                avg_fps = sum(fps_values) / len(fps_values)
                time_per_triangle = avg_total_time / total_triangles
                # Format the numbers to reasonable decimal places
                avg_total_time = round(avg_total_time, 4)
                avg_fps = round(avg_fps, 4)
                time_per_triangle = round(time_per_triangle, 8)
                # Update the profile data
                profile['total_time'] = str(avg_total_time)
                profile['fps'] = str(avg_fps)
                profile['total_triangles'] = str(total_triangles)
                profile['time_per_triangle'] = str(time_per_triangle)
                # Update the timings.md file
                timings_md = update_timings_md(timings_md, profile, avg_total_time, avg_fps, total_triangles, time_per_triangle)
            else:
                print(f"Could not compute averages for profile {profile['name']}.")
        else:
            # Convert string values to appropriate types
            profile['total_time'] = float(profile['total_time'])
            profile['fps'] = float(profile['fps'])
            profile['total_triangles'] = int(profile['total_triangles'])
            profile['time_per_triangle'] = float(profile['time_per_triangle'])

    # Write the updated timings.md file
    with open('Bonus1/timings.md', 'w') as f:
        f.write(timings_md)
    print("\nUpdated timings.md successfully.")

    # Plot the metrics
    plot_metrics(profiles)

if __name__ == '__main__':
    main()

