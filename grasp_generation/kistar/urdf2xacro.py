import re

def extract_unique_values(urdf_text, pattern):
    """특정 XML 패턴에서 중복되지 않는 수치값만 뽑아낸다."""
    values = re.findall(pattern, urdf_text)
    return sorted(set(values), key=values.index)

def replace_with_xacro_property(urdf_text, pattern, var_prefix):
    values = extract_unique_values(urdf_text, pattern)
    prop_lines = []
    replaced_text = urdf_text

    for idx, value in enumerate(values):
        var_name = f"{var_prefix}_{idx+1}"
        # xacro property 선언
        prop_lines.append(f'  <xacro:property name="{var_name}" value="{value}"/>')
        # 패턴 대체
        replaced_text = replaced_text.replace(value, f"${{{var_name}}}")

    return "\n".join(prop_lines), replaced_text

def urdf_to_xacro(urdf_file, xacro_file):
    with open(urdf_file, "r") as f:
        urdf = f.read()

    # 1. 헤더 변환
    urdf = re.sub(r'<robot ', '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" ', urdf, count=1)

    # 2. 각종 수치값 패턴 추출 (필요시 추가 가능)
    # box size
    box_prop, urdf = replace_with_xacro_property(urdf, r'\d+\.\d+ \d+\.\d+ \d+\.\d+', "box_size")
    # mass
    mass_prop, urdf = replace_with_xacro_property(urdf, r'(?<=mass value=")\d+\.\d+', "mass")
    # inertia
    inertia_prop, urdf = replace_with_xacro_property(urdf, r'(?<=inertia ixx=")\d+\.\d+', "ixx")

    # 3. property 선언부 파일 최상단에 추가
    prop_section = "\n".join([box_prop, mass_prop, inertia_prop])
    urdf = re.sub(r'(<robot [^\n]*>)', r'\1\n' + prop_section, urdf, count=1)

    # 4. 저장
    with open(xacro_file, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(urdf)

    print(f"변환 완료! 저장 위치: {xacro_file}")

# 사용법 예시
# urdf_to_xacro("input.urdf", "output.xacro")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python urdf_to_xacro.py input.urdf output.xacro")
    else:
        urdf_to_xacro(sys.argv[1], sys.argv[2])
