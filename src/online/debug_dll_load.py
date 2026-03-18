import os
import sys
import ctypes
from pathlib import Path

def test_dlls():
    print("=== 底层 DLL 强行唤醒诊断 ===")
    
    # 直接定位到打包后的 torch/lib 案发现场
    project_root = Path(__file__).resolve().parents[2]
    internal_dir = project_root / "dist" / "main_app" / "_internal"
    target_dir = internal_dir / "torch" / "lib"
    
    if not target_dir.exists():
        print(f"❌ 找不到目录: {target_dir}")
        return

    # 1. 模拟最高级别的环境变量和目录优先级
    os.chdir(str(target_dir))
    print(f"📂 已强行切入目录: {target_dir}")
    os.environ['PATH'] = str(internal_dir) + os.pathsep + str(target_dir) + os.pathsep + os.environ.get('PATH', '')
    
    if sys.version_info >= (3, 8):
        os.add_dll_directory(str(internal_dir))
        os.add_dll_directory(str(target_dir))

    # 2. 抓取所有 DLL
    dlls = [f for f in os.listdir('.') if f.endswith('.dll')]
    print(f"🔍 发现 {len(dlls)} 个 DLL，开始执行 Windows 底层加载测试...\n")

    failed_dlls = {}
    passed_count = 0

    # 3. 逐个进行 ctypes 级加载，抓取最真实的报错
    for dll in dlls:
        try:
            # WinDLL 会直接调用 Windows 的 LoadLibraryEx
            ctypes.WinDLL(os.path.abspath(dll))
            passed_count += 1
        except OSError as e:
            failed_dlls[dll] = str(e)
        except Exception as e:
            failed_dlls[dll] = f"未知错误: {str(e)}"

    print("-" * 50)
    print(f"✅ 成功加载: {passed_count} 个")
    print(f"❌ 失败加载: {len(failed_dlls)} 个")
    
    if failed_dlls:
        print("\n🚨 崩溃报告（注意看报错代码）：")
        for dll, err in failed_dlls.items():
            print(f"  [{dll}] => {err}")
            
        if 'c10.dll' in failed_dlls and '1114' in failed_dlls['c10.dll']:
            print("\n💡 诊断结论：")
            print("c10.dll 的 1114 错误被复现！这意味着它不是被 python import 连累的，而是它自身的 C++ 编译环境和你当前系统的依赖发生了极其深度的不兼容。")
    else:
        print("\n🤔 活见鬼了，所有 DLL 都能底层独立加载成功！说明文件全都是好的，纯粹是 PyInstaller 运行时的上下文污染了它。")

if __name__ == '__main__':
    test_dlls()
