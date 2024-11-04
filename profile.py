import subprocess
import re

def get_profiles(timings_md):
    pattern = r'### (\w+)\s+File: (.+?)\s+Description: (.+?)\s+Average Time Total:(.*?)\s+Average Time FPS:(.*?)\s+(?=###|\Z)'
    matches = re.findall(pattern, timings_md, re.DOTALL)
    profiles = []
    for match in matches:
        profile_name = match[0]
        total_time = match[3].strip()
        fps = match[4].strip()
        profiles.append({
            'name': profile_name,
            'total_time': total_time,
            'fps': fps,
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

def run_profile(executable):
    print(f"Running executable: ./{executable}")
    result = subprocess.run([f'./{executable}'], capture_output=True, text=True)
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

def update_timings_md(timings_md, profile, avg_total_time, avg_fps):
    # Build the replacement section
    section = f"""### {profile['name']}

File: {profile['section'][1]}
Description: {profile['section'][2]}
Average Time Total: {avg_total_time}
Average Time FPS: {avg_fps}
"""
    pattern = re.compile(rf'### {profile["name"]}\s+File: .+?\s+Description: .+?\s+Average Time Total:.*?\s+Average Time FPS:.*?(?=###|\Z)', re.DOTALL)
    timings_md = pattern.sub(section.strip(), timings_md)
    return timings_md

def main():
    with open('./Bonus1/timings.md', 'r') as f:
        timings_md = f.read()

    profiles = get_profiles(timings_md)

    for profile in profiles:
        if profile['total_time'] == '' or profile['fps'] == '':
            print(f"\nProcessing profile: {profile['name']}")
            # Build the profile
            if not build_profile(profile['name']):
                continue
            # Run the profile 10 times
            total_times = []
            fps_values = []
            executable = profile['name'].lower()
            for i in range(10):
                print(f"\nRun {i+1}/10 for profile {profile['name']}")
                result = run_profile(executable)
                if result:
                    total_time, fps = result
                    total_times.append(total_time)
                    fps_values.append(fps)
                else:
                    print(f"Skipping run {i+1} due to errors.")
            if total_times and fps_values:
                avg_total_time = sum(total_times) / len(total_times)
                avg_fps = sum(fps_values) / len(fps_values)
                # Update the timings.md file
                timings_md = update_timings_md(timings_md, profile, avg_total_time, avg_fps)

    # Write the updated timings.md file
    with open('./Bonus1/timings.md', 'w') as f:
        f.write(timings_md)
    print("\nUpdated timings.md successfully.")

if __name__ == '__main__':
    main()

