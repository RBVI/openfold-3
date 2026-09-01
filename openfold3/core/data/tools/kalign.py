# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache



@lru_cache(maxsize=512)
def _run_kalign_cached(sequences: tuple[str, ...]) -> str:
    """Wrapper around kalign.align with caching."""
    from sys import platform
    if platform == 'win32' or platform == 'darwin':
        # The Python kalign module is not available on Windows.
        # The Python kalign module aborts on Mac due to duplicate libomp.dylib.
        return _run_kalign_exe(sequences)
    else:
        import kalign
        return kalign.align(list(sequences))


def run_kalign(
    sequences: list[str],
) -> list[str]:
    """Runs Kalign on the provided A3M string and returns the aligned sequences.

    Args:
        sequences (list[str]):
            Sequences to be aligned. In the template pipeline,
            the first sequence is the query, and the rest are templates
            sequences to be realigned to it from hmmsearch.
    Returns:
        list:
            The aligned sequences as a list of strings.
    """
    return _run_kalign_cached(tuple(sequences))

def _run_kalign_exe(
    sequences: list[str],
) -> list[str]:
    """
    Runs the kalign executable in a subprocess and returns the aligned sequences.
    This is used on Windows because there is currently (March 2026) no PyPi kalign
    package for Windows.

    Args:
        sequences (list[str]):
            Sequences to be aligned. In the template pipeline,
            the first sequence is the query, and the rest are templates
            sequences to be realigned to it from hmmsearch.
    Returns:
        list:
            The aligned sequences as a list of strings.
    """
    import sys
    from os.path import dirname, join, exists
    kalign_exe = join(dirname(sys.executable), 'kalign')
    if not exists(kalign_exe):
        kalign_exe = shutil.which("kalign")

    if kalign_exe is None:
        raise RuntimeError(
            "Kalign is not available. Please install it and ensure it is in your PATH."
        )

    a3m_string = '\n'.join(f'>sequence_{i}\n{seq}' for i,seq in enumerate(sequences))
    import subprocess
    try:
        result = subprocess.run(
            [kalign_exe], input=a3m_string, capture_output=True, text=True, check=True
        )

        # The resulting MSA is stored in the variable
        alignment_result = result.stdout

        # Strip header from kalign version 3.
        if not alignment_result.startswith('>') and '\n>' in alignment_result:
            alignment_result = alignment_result[alignment_result.find('\n>'):]

    except subprocess.CalledProcessError as e:
        print(f"Kalign command failed:\n{e.stderr}")

    # Convert fasta/a3m to list of sequences.
    aligned_seqs = []
    seq = ''
    for line in alignment_result.split('\n'):
        if line.startswith('>'):
            if seq:
                aligned_seqs.append(seq)
                seq = ''
        else:
            seq += line
    aligned_seqs.append(seq)

    return aligned_seqs
